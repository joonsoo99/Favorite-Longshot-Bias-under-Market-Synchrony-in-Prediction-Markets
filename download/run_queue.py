# -*- coding: utf-8 -*-
# Sequential queue: 60m -> 15m -> 5m. 60m is assumed to already be running in
# a separate process, so this script only polls until it's done, then runs
# 15m and 5m in-process (each run() blocks until its own jobs finish, so
# they execute in order naturally). Usage: python run_queue.py
import time
import logging

import pandas as pd

import collect_polymarket_updown as m

log = logging.getLogger("run_queue")


# Count of jobs never attempted (success or permanent failure) for this
# timeframe. Checking existing_keys (successes) alone is not enough: a slot
# that permanently failed this run (e.g. 79 such slots for 60m) never enters
# the success list, so a success-only check would poll forever. Slots already
# logged to the missing pkl count as "attempted" and are excluded from "todo".
def pending_count(tf: str) -> int:
    cfg = m.TF_REGISTRY[tf]
    start_epoch = m.floor_to_epoch(m.dt_to_epoch(m.parse_utc(cfg.start_utc)), cfg.interval_sec)
    end_epoch   = m.floor_to_epoch(m.dt_to_epoch(m.parse_utc(cfg.end_utc)), cfg.interval_sec) - cfg.interval_sec
    slots = list(range(start_epoch, end_epoch + cfg.interval_sec, cfg.interval_sec))

    pkl_path, missing_path = m.out_paths(tf)
    _, existing_keys = m.load_existing(pkl_path)

    attempted_keys = set(existing_keys)
    if missing_path.exists():
        try:
            df_miss = pd.read_pickle(missing_path)
            attempted_keys |= set(zip(df_miss["asset"], df_miss["slot_epoch"]))
        except Exception as e:
            log.warning(f"[queue] failed to load missing-log (ignoring, continuing): {e}")

    all_jobs = [(a, s) for s in slots for a in cfg.assets if s not in cfg.missing_slots]
    todo = [(a, s) for a, s in all_jobs if (cfg.asset_labels[a], s) not in attempted_keys]
    return len(todo)


def wait_until_done(tf: str, poll_sec: int = 60) -> None:
    """Poll (don't run) until another process finishes collecting `tf`."""
    while True:
        n = pending_count(tf)
        if n == 0:
            log.info(f"[queue] {tf} confirmed done (0 jobs remaining)")
            return
        log.info(f"[queue] {tf} still has {n:,} jobs left → rechecking in {poll_sec}s")
        time.sleep(poll_sec)


def main():
    log.info("[queue] waiting for 60m to finish (running in another process)")
    wait_until_done("60m")

    log.info("[queue] starting 15m")
    m.run("15m")
    log.info("[queue] 15m done")

    log.info("[queue] starting 5m")
    m.run("5m")
    log.info("[queue] 5m done")

    log.info("[queue] full queue done (60m→15m→5m)")


if __name__ == "__main__":
    main()
