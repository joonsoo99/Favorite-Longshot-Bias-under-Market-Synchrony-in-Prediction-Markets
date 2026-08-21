from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("collect")

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "raw_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("America/New_York")

GAMMA_EVENT_BY_SLUG  = "https://gamma-api.polymarket.com/events/slug"
GAMMA_MARKET_BY_SLUG = "https://gamma-api.polymarket.com/markets/slug"
CLOB_PRICE_HISTORY   = "https://clob.polymarket.com/prices-history"

HTTP_TIMEOUT   = 12
HTTP_RETRIES   = 3
HTTP_BACKOFF_S = 0.5


def parse_utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def dt_to_epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def floor_to_epoch(ts: int, interval_sec: int) -> int:
    return ts - (ts % interval_sec)


def epoch_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# Key = minutes remaining until close (NOT elapsed since open) — this convention
# must be uniform across all timeframes to avoid reversing ttm chronology.
def minute_left_dict(end_ts: int, history: list, window_minutes: int) -> dict[int, float]:
    d: dict[int, float] = {}
    for h in history:
        m_left = int((end_ts - h["t"]) // 60)
        if 0 <= m_left <= window_minutes:
            d[m_left] = h["p"]
    return d


def _get_with_retry(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    last_exc = None
    for attempt in range(HTTP_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(HTTP_BACKOFF_S * (2 ** attempt) * 3)
                continue
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(HTTP_BACKOFF_S * (2 ** attempt))
    if last_exc is not None:
        log.debug(f"HTTP retry failed after {HTTP_RETRIES} attempts: {url} ({last_exc})")
    return None


def fetch_event_meta(slug: str) -> tuple[Optional[dict], int]:
    r = _get_with_retry(f"{GAMMA_EVENT_BY_SLUG}/{slug}")
    if r is None:
        return None, -1
    if r.status_code != 200:
        return None, r.status_code
    return r.json(), 200


def fetch_market_meta(slug: str) -> tuple[Optional[dict], int]:
    r = _get_with_retry(f"{GAMMA_MARKET_BY_SLUG}/{slug}")
    if r is None:
        return None, -1
    if r.status_code != 200:
        return None, r.status_code
    return r.json(), 200


def fetch_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int = 1) -> Optional[list]:
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity}
    r = _get_with_retry(CLOB_PRICE_HISTORY, params=params)
    if r is None or r.status_code != 200:
        return None
    return r.json().get("history", [])


# Explicit failure instead of silently defaulting to index 0 when "Up" is missing.
def resolve_yes_no_index(outcomes: list, slug: str) -> Optional[tuple[int, int]]:
    if outcomes is None:
        log.warning(f"outcomes is None: slug={slug}")
        return None
    if len(outcomes) != 2:
        log.warning(f"outcomes has != 2 entries (violates binary-market assumption): slug={slug} outcomes={outcomes}")
        return None
    if "Up" in outcomes:
        yes_idx = outcomes.index("Up")
    else:
        lower_map = {str(o).lower(): i for i, o in enumerate(outcomes)}
        if "up" in lower_map:
            yes_idx = lower_map["up"]
            log.warning(f"'Up' case variant detected (recovered): slug={slug} outcomes={outcomes}")
        else:
            log.warning(f"no 'Up' in outcomes, skipping slot: slug={slug} outcomes={outcomes}")
            return None
    no_idx = 1 - yes_idx
    return yes_idx, no_idx


@dataclass
class TFConfig:
    name: str
    window_minutes: int
    interval_sec: int
    assets: list[str]
    asset_labels: dict[str, str]
    max_workers: int
    meta_kind: str
    slug_fn: Callable[[str, int], str]
    start_utc: str
    end_utc: str
    missing_slots: set[int] = field(default_factory=set)


def slug_5m(asset: str, slot_epoch: int) -> str:
    return f"{asset}-updown-5m-{slot_epoch}"


def slug_15m(asset: str, slot_epoch: int) -> str:
    return f"{asset}-updown-15m-{slot_epoch}"


# Polymarket changed the 60m slug format (year present/absent) twice over the
# collection window; these boundaries were reverse-engineered from live 404s.
_SLUG_YEAR_EPOCH    = 1773550800
_SLUG_NOYEAR_EPOCH  = 1774472400
_SLUG_YEAR2_EPOCH   = 1774479600
_SLUG_NOYEAR2_EPOCH = 1775188800
_SLUG_YEAR3_EPOCH   = 1775239200


def slug_60m(asset: str, slot_epoch: int) -> str:
    dt_utc = datetime.fromtimestamp(slot_epoch, tz=timezone.utc)
    dt_et  = dt_utc.astimezone(ET)
    month_name = dt_et.strftime("%B").lower()
    day  = dt_et.day
    hour = dt_et.hour
    if hour == 0:
        hour_str = "12am"
    elif hour < 12:
        hour_str = f"{hour}am"
    elif hour == 12:
        hour_str = "12pm"
    else:
        hour_str = f"{hour - 12}pm"

    use_year = (
        (_SLUG_YEAR_EPOCH  <= slot_epoch < _SLUG_NOYEAR_EPOCH) or
        (_SLUG_YEAR2_EPOCH <= slot_epoch < _SLUG_NOYEAR2_EPOCH) or
        (slot_epoch >= _SLUG_YEAR3_EPOCH)
    )
    if use_year:
        return f"{asset}-up-or-down-{month_name}-{day}-{dt_et.year}-{hour_str}-et"
    return f"{asset}-up-or-down-{month_name}-{day}-{hour_str}-et"


# Confirmed via repeated 404s: these 60m markets were never actually opened.
_HOURLY_MISSING_SLOTS = {
    1774472400, 1775188800, 1775192400, 1775196000, 1775199600,
    1775203200, 1775206800, 1775210400, 1775232000, 1775235600,
}


TF_REGISTRY: dict[str, TFConfig] = {
    "5m": TFConfig(
        name="5m", window_minutes=5, interval_sec=5 * 60,
        assets=["btc", "eth", "sol", "xrp"],
        asset_labels={"btc": "bitcoin", "eth": "ethereum", "sol": "solana", "xrp": "xrp"},
        max_workers=32, meta_kind="event", slug_fn=slug_5m,
        start_utc="2026-02-19 00:00:00", end_utc="2026-07-31 00:00:00",
    ),
    "15m": TFConfig(
        name="15m", window_minutes=15, interval_sec=15 * 60,
        assets=["btc", "eth", "sol", "xrp"],
        asset_labels={"btc": "btc", "eth": "eth", "sol": "sol", "xrp": "xrp"},
        max_workers=16, meta_kind="market", slug_fn=slug_15m,
        start_utc="2025-11-01 00:00:00", end_utc="2026-07-31 00:00:00",
    ),
    "60m": TFConfig(
        name="60m", window_minutes=60, interval_sec=60 * 60,
        assets=["bitcoin", "ethereum", "solana", "xrp"],
        asset_labels={"bitcoin": "bitcoin", "ethereum": "ethereum", "solana": "solana", "xrp": "xrp"},
        max_workers=8, meta_kind="market", slug_fn=slug_60m,
        start_utc="2025-07-01 00:00:00", end_utc="2026-07-31 00:00:00",
        missing_slots=set(_HOURLY_MISSING_SLOTS),
    ),
}


def out_paths(tf: str) -> tuple[Path, Path]:
    return OUT_DIR / f"{tf}_poly_raw.pkl", OUT_DIR / f"{tf}_poly_missing.pkl"


def load_existing(pkl_path: Path) -> tuple[pd.DataFrame, set]:
    if pkl_path.exists():
        try:
            df = pd.read_pickle(pkl_path)
            existing = set(zip(df["asset"], df["slot_epoch"]))
            log.info(f"[Load] loaded {len(df):,} existing rows ({len(existing):,} (asset,slot) pairs)")
            return df, existing
        except Exception as e:
            log.warning(f"[Load] failed to load pkl: {e} → starting empty")
    return pd.DataFrame(), set()


# 5m markets are nested under /events/slug (markets[0]); 15m/60m are
# fetched directly from /markets/slug — both are normalized into `meta`.
def collect_meta_one_slot(cfg: TFConfig, asset_code: str, slot_epoch: int):
    slug = cfg.slug_fn(asset_code, slot_epoch)
    base = {
        "asset": cfg.asset_labels[asset_code],
        "slot_epoch": slot_epoch,
        "slot_utc": epoch_to_iso(slot_epoch),
        "slug": slug,
    }

    if cfg.meta_kind == "event":
        event_data, status = fetch_event_meta(slug)
        if event_data is None:
            return False, {**base, "status": status, "reason": "http_failed"}
        markets = event_data.get("markets")
        if not markets:
            return False, {**base, "status": 200, "reason": "no_markets_in_event"}
        meta = markets[0]
    else:
        meta, status = fetch_market_meta(slug)
        if meta is None:
            return False, {**base, "status": status, "reason": "http_failed"}

    clobs = meta.get("clobTokenIds")
    if clobs is None:
        return False, {**base, "status": 200, "reason": "no_clobTokenIds"}
    try:
        volume = float(meta["volume"]) if meta.get("volume") is not None else None
    except (TypeError, ValueError):
        volume = None

    return True, {**base, "clobTokenIds": clobs,
                   "outcomes": meta.get("outcomes"), "outcomePrices": meta.get("outcomePrices"),
                   "volume": volume}


def collect_price_history_one_row(cfg: TFConfig, row: dict):
    slug = row.get("slug", "?")
    try:
        clobs          = json.loads(row["clobTokenIds"]) if isinstance(row["clobTokenIds"], str) else row["clobTokenIds"]
        outcomes       = json.loads(row["outcomes"])      if isinstance(row["outcomes"], str)      else row["outcomes"]
        outcome_prices = json.loads(row["outcomePrices"]) if isinstance(row["outcomePrices"], str) else row["outcomePrices"]

        idx_pair = resolve_yes_no_index(outcomes, slug)
        if idx_pair is None:
            return False, {**row, "reason": "outcomes_missing_up_label"}
        yes_idx, no_idx = idx_pair
        token_yes, token_no = clobs[yes_idx], clobs[no_idx]

        # close = open (slot_epoch) + window; fetch exactly that window's history.
        end_ts   = int(row["slot_epoch"]) + cfg.window_minutes * 60
        start_ts = end_ts - cfg.window_minutes * 60

        hist_yes = fetch_price_history(token_yes, start_ts, end_ts)
        hist_no  = fetch_price_history(token_no, start_ts, end_ts)
        if hist_yes is None or hist_no is None:
            return False, {**row, "reason": "price_history_fetch_failed"}
        if not hist_yes or not hist_no:
            return False, {**row, "reason": "empty_price_history"}

        dict_yes = minute_left_dict(end_ts, hist_yes, cfg.window_minutes)
        dict_no  = minute_left_dict(end_ts, hist_no, cfg.window_minutes)

        # outcomePrices is read at meta-fetch time, not guaranteed final — only
        # trust it when the price is confidently settled (near 0 or 1); otherwise
        # this slot hasn't resolved yet and must not be labeled.
        p_yes = float(outcome_prices[yes_idx])
        if not (p_yes <= 0.02 or p_yes >= 0.98):
            return False, {**row, "reason": "market_not_resolved"}
        outcome = int(p_yes > 0.5)

        return True, {**row, "price_history_yes": dict_yes, "price_history_no": dict_no,
                       "outcome": outcome}
    except Exception as e:
        return False, {**row, "reason": f"exception:{e}"}


def save_merged(pkl_path: Path, df_existing: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    if df_new.empty:
        log.info("[Save] no new data, skipping save")
        return df_existing
    df_out = df_new if df_existing.empty else (
        pd.concat([df_existing, df_new], ignore_index=True)
          .drop_duplicates(subset=["asset", "slot_epoch"])
          .reset_index(drop=True)
    )
    df_out = df_out.sort_values(["asset", "slot_epoch"]).reset_index(drop=True)
    df_out.to_pickle(pkl_path)
    log.info(f"[Save] saved → {pkl_path} (total {len(df_out):,} rows, {len(df_new):,} new)")
    return df_out


# Saved incrementally after every chunk so a crash mid-run loses at most one chunk.
CHUNK_SIZE = 5000


def _save_missing(missing_path: Path, miss_rows: list[dict]) -> None:
    if not miss_rows:
        return
    if missing_path.exists():
        try:
            prev = pd.read_pickle(missing_path)
            miss_rows = pd.concat([prev, pd.DataFrame(miss_rows)], ignore_index=True).to_dict("records")
        except Exception:
            pass
    pd.DataFrame(miss_rows).to_pickle(missing_path)


def _process_chunk(cfg: TFConfig, chunk_jobs: list[tuple[str, int]],
                    df_existing: pd.DataFrame, pkl_path: Path, missing_path: Path) -> pd.DataFrame:
    meta_ok, meta_miss = [], []
    with tqdm(total=len(chunk_jobs), desc="  Step1 Meta", leave=False) as pbar:
        with ThreadPoolExecutor(cfg.max_workers) as ex:
            futs = {ex.submit(collect_meta_one_slot, cfg, a, s): (a, s) for a, s in chunk_jobs}
            for fut in as_completed(futs):
                ok, res = fut.result()
                (meta_ok if ok else meta_miss).append(res)
                pbar.update(1)

    price_ok, price_miss = [], []
    if meta_ok:
        df_meta = pd.DataFrame(meta_ok)
        with tqdm(total=len(df_meta), desc="  Step2 Price", leave=False) as pbar:
            with ThreadPoolExecutor(cfg.max_workers) as ex:
                futs = {ex.submit(collect_price_history_one_row, cfg, r): r for r in df_meta.to_dict("records")}
                for fut in as_completed(futs):
                    ok, res = fut.result()
                    (price_ok if ok else price_miss).append(res)
                    pbar.update(1)

    log.info(f"  chunk result: meta ok {len(meta_ok):,}/fail {len(meta_miss):,}  "
              f"→ price ok {len(price_ok):,}/fail {len(price_miss):,}")

    _save_missing(missing_path, meta_miss + price_miss)

    df_new = pd.DataFrame(price_ok) if price_ok else pd.DataFrame()
    return save_merged(pkl_path, df_existing, df_new)


# Resumable: skips (asset, slot_epoch) pairs already present in the pkl.
def run(tf: str) -> None:
    cfg = TF_REGISTRY[tf]
    pkl_path, missing_path = out_paths(tf)

    start_epoch = floor_to_epoch(dt_to_epoch(parse_utc(cfg.start_utc)), cfg.interval_sec)
    end_epoch   = floor_to_epoch(dt_to_epoch(parse_utc(cfg.end_utc)), cfg.interval_sec) - cfg.interval_sec
    slots = list(range(start_epoch, end_epoch + cfg.interval_sec, cfg.interval_sec))

    log.info("=" * 60)
    log.info(f"Polymarket {tf} Collector")
    log.info("=" * 60)
    log.info(f"Start UTC : {datetime.fromtimestamp(start_epoch, tz=timezone.utc)}")
    log.info(f"End UTC   : {datetime.fromtimestamp(end_epoch, tz=timezone.utc)}")
    log.info(f"Slots     : {len(slots):,}  Assets: {cfg.assets}")

    df_existing, existing_keys = load_existing(pkl_path)

    all_jobs = [(a, s) for s in slots for a in cfg.assets
                if s not in cfg.missing_slots]
    todo_jobs = [(a, s) for a, s in all_jobs
                 if (cfg.asset_labels[a], s) not in existing_keys]

    log.info(f"[Filter] {len(all_jobs):,} total, skipping {len(existing_keys):,} existing → {len(todo_jobs):,} new")

    if not todo_jobs:
        log.info("[Done] no new slots to collect")
        return

    n_chunks = (len(todo_jobs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(todo_jobs), CHUNK_SIZE):
        chunk = todo_jobs[i:i + CHUNK_SIZE]
        chunk_no = i // CHUNK_SIZE + 1
        log.info(f"[{tf}] chunk {chunk_no}/{n_chunks} ({len(chunk):,} jobs)...")
        df_existing = _process_chunk(cfg, chunk, df_existing, pkl_path, missing_path)

    log.info(f"[{tf}] all done ({n_chunks} chunks total)")


def main():
    parser = argparse.ArgumentParser(description="Polymarket UP/DOWN market collector")
    parser.add_argument("--tf", choices=["5m", "15m", "60m", "all"], required=True)
    args = parser.parse_args()

    targets = ["5m", "15m", "60m"] if args.tf == "all" else [args.tf]
    for tf in targets:
        run(tf)


if __name__ == "__main__":
    main()
