# -*- coding: utf-8 -*-
"""
run_queue.py — 시간대 순차 실행 큐

60m은 이미 별도 프로세스로 실행 중이므로, 이 스크립트는 60m을 다시
실행하지 않고 "남은 작업이 0인지"만 주기적으로 확인(폴링)한다.
60m이 끝난 게 확인되면 15m → 5m을 이 프로세스 안에서 순서대로 실행한다
(각 run()은 자기 몫이 전부 끝날 때까지 블로킹되므로 자연히 순차 진행됨).

사용법: python run_queue.py
"""
import time
import logging

import pandas as pd

import collect_polymarket_updown as m

log = logging.getLogger("run_queue")


def pending_count(tf: str) -> int:
    """해당 시간대의 '아직 한 번도 시도 안 된' 작업 수. 0이면 (성공이든 영구실패든)
    전부 시도가 끝났다는 뜻이다.

    주의: existing_keys(성공)만 보면 안 된다 — 이번 실행에서 영구 실패한
    슬롯(예: 60m의 79건)은 성공 목록엔 절대 안 들어오므로, 성공 여부만
    기준으로 삼으면 폴링이 끝나지 않는다. missing 로그에 이미 기록된
    (attempted) 슬롯도 "시도 완료"로 쳐서 대기 목록에서 제외한다.
    """
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
            log.warning(f"[queue] missing 로그 로드 실패(무시하고 진행): {e}")

    all_jobs = [(a, s) for s in slots for a in cfg.assets if s not in cfg.missing_slots]
    todo = [(a, s) for a, s in all_jobs if (cfg.asset_labels[a], s) not in attempted_keys]
    return len(todo)


def wait_until_done(tf: str, poll_sec: int = 60) -> None:
    """다른 프로세스가 처리 중인 tf가 끝날 때까지 폴링만 한다 (직접 실행하지 않음)."""
    while True:
        n = pending_count(tf)
        if n == 0:
            log.info(f"[queue] {tf} 완료 확인됨 (남은 작업 0건)")
            return
        log.info(f"[queue] {tf} 아직 {n:,}건 남음 → {poll_sec}초 후 재확인")
        time.sleep(poll_sec)


def main():
    log.info("[queue] 60m 완료 대기 시작 (60m 자체는 다른 프로세스가 처리 중)")
    wait_until_done("60m")

    log.info("[queue] 15m 시작")
    m.run("15m")
    log.info("[queue] 15m 완료")

    log.info("[queue] 5m 시작")
    m.run("5m")
    log.info("[queue] 5m 완료")

    log.info("[queue] 전체 큐 완료 (60m→15m→5m)")


if __name__ == "__main__":
    main()
