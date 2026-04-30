"""
메인 오케스트레이터

사용법:
    python main.py backfill      # 2025-07-01 ~ 어제 전체 생성
    python main.py incremental   # 어제 데이터만 생성/갱신

실행 순서 (테이블 의존성 순):
  1. users              (독립)
  2. unit_stats         (독립)
  3. pilot_stats        (독립)
  4. pilot_unit_synergy (unit + pilot 의존)
  5. gacha_event_log    (unit 의존)
  6. session_log        (users 의존)          ─┐
  7. gacha_history      (users + unit 의존)    ├─ daily_raw
  8. stage_clear_log    (users 의존)           │
  9. stage_unit_log     (stage_clear 의존)     │
 10. party_composition  (stage_clear 의존)    ─┘
 11. user_unit_inventory (gacha_history 의존)
 12. events             (users 의존)           ─┐
 13. purchases          (users 의존)            ├─ analytics
 14. marketing_events   (users 의존)           ─┘
 15. user_daily_snapshot (events + purchases 의존)
 16. daily_game_metrics  (events + purchases 의존)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# data_generator 패키지 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

from config import START_DATE, YESTERDAY, TODAY, OUTPUT_DIR


def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run(mode: str = "backfill") -> None:
    assert mode in ("backfill", "incremental"), \
        f"mode는 'backfill' 또는 'incremental'이어야 합니다. 입력값: {mode}"

    _banner(f"Game Revenue Data Generator  |  mode={mode}")
    print(f"  기간: {START_DATE} ~ {YESTERDAY}  (오늘: {TODAY})")
    print(f"  출력: {OUTPUT_DIR}\n")
    total_start = time.time()

    # ── Step 1: users ──────────────────────────────────────────────────────
    _banner("Step 1 / 6 : users")
    import generate_users
    users_df = generate_users.run(mode)

    # ── Step 2-5: meta (unit, pilot, synergy, gacha_event_log) ────────────
    _banner("Step 2 / 6 : meta tables (unit_stats, pilot_stats, synergy, gacha_event_log)")
    import generate_meta
    generate_meta.run(mode)

    # ── Step 3: daily raw logs ─────────────────────────────────────────────
    _banner("Step 3 / 6 : daily raw logs (session, gacha, stage, party)")
    import generate_daily_raw
    generate_daily_raw.run(mode, users_df=users_df)

    # ── Step 4: user_unit_inventory ────────────────────────────────────────
    _banner("Step 4 / 6 : user_unit_inventory")
    import generate_inventory
    generate_inventory.run(mode)

    # ── Step 5-7: analytics (events, purchases, marketing_events) ─────────
    _banner("Step 5 / 6 : analytics tables (events, purchases, marketing_events)")
    import generate_analytics
    generate_analytics.run(mode, users_df=users_df)

    # ── Step 6-7: snapshots (user_daily_snapshot, daily_game_metrics) ─────
    _banner("Step 6 / 6 : snapshots (user_daily_snapshot, daily_game_metrics)")
    import generate_snapshots
    generate_snapshots.run(mode)

    elapsed = time.time() - total_start
    _banner(f"완료  |  총 소요 시간: {elapsed:.1f}초")
    print(f"  데이터 출력 경로: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "backfill"
    run(mode)
