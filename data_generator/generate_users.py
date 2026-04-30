"""
테이블 1: users
갱신 유형: Append (신규 가입) + Update (last_login_date)

backfill: 전체 30,000명 한 번에 생성
incremental: last_login_date만 갱신 (이탈 유저 제외)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

from config import (
    START_DATE, TODAY, YESTERDAY, N_USERS, RANDOM_SEED,
    COUNTRIES, COUNTRY_WEIGHTS, DEVICES, DEVICE_WEIGHTS,
    CHANNELS, CHANNEL_WEIGHTS, CHURN_RATE, DAU_RATE, OUTPUT_DIR,
)

OUT_FILE = OUTPUT_DIR / "raw" / "users" / "users.parquet"


def generate_users(rng: np.random.Generator) -> pd.DataFrame:
    """30,000명 유저 기본 정보를 한 번에 생성."""
    days_total = (YESTERDAY - START_DATE).days + 1

    # signup_date: 초반에 더 많은 신규 유저, 이후 점진적 감소
    day_indices = np.arange(days_total)
    signup_weights = np.exp(-day_indices / (days_total * 0.6)) + 0.2
    signup_weights /= signup_weights.sum()
    signup_day_offsets = rng.choice(day_indices, size=N_USERS, p=signup_weights)
    signup_dates = [START_DATE + timedelta(days=int(d)) for d in signup_day_offsets]

    countries  = rng.choice(COUNTRIES, size=N_USERS, p=COUNTRY_WEIGHTS)
    devices    = rng.choice(DEVICES,   size=N_USERS, p=DEVICE_WEIGHTS)
    channels   = rng.choice(CHANNELS,  size=N_USERS, p=CHANNEL_WEIGHTS)

    acq_costs = np.where(
        channels == "paid",
        np.round(rng.uniform(0.5, 5.0, N_USERS), 2),
        0.0,
    )

    ab_groups = rng.choice(["A", "B"], size=N_USERS)

    # 이탈 여부 결정
    churn_mask = rng.random(N_USERS) < CHURN_RATE
    # 이탈까지 걸리는 일수: 로그-정규 분포 (평균 ~21일, 꼬리 길게)
    churn_survival_days = np.clip(
        rng.lognormal(mean=2.5, sigma=1.0, size=N_USERS).astype(int), 1, 180
    )

    first_login_dates = []
    last_login_dates  = []

    for i in range(N_USERS):
        signup = signup_dates[i]
        first_login = signup + timedelta(days=int(rng.integers(0, 2)))
        first_login = min(first_login, YESTERDAY)
        first_login_dates.append(first_login)

        if churn_mask[i]:
            last_login = signup + timedelta(days=int(churn_survival_days[i]))
            last_login = min(last_login, YESTERDAY)
            last_login = max(last_login, first_login)
        else:
            # 활성 유저: 최근 7일 이내 로그인
            days_ago = int(rng.integers(0, 7))
            last_login = max(YESTERDAY - timedelta(days=days_ago), first_login)

        last_login_dates.append(last_login)

    df = pd.DataFrame({
        "user_id":             [f"U{i+1:06d}" for i in range(N_USERS)],
        "signup_date":         signup_dates,
        "first_login_date":    first_login_dates,
        "last_login_date":     last_login_dates,
        "country":             countries,
        "device_type":         devices,
        "acquisition_channel": channels,
        "acquisition_cost":    acq_costs,
        "ab_group":            ab_groups,
    })

    return df


def run(mode: str = "backfill") -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if mode == "backfill" or not OUT_FILE.exists():
        print("[users] backfill: 30,000명 생성 중...")
        df = generate_users(rng)
        df.to_parquet(OUT_FILE, index=False)
        print(f"[users] 저장 완료 → {OUT_FILE}  ({len(df):,}행)")
    else:
        # incremental: last_login_date가 오늘인 활성 유저 갱신
        print("[users] incremental: last_login_date 갱신 중...")
        df = pd.read_parquet(OUT_FILE)
        active_mask = df["last_login_date"] >= (YESTERDAY - timedelta(days=7))
        # 활성 유저 중 일부가 오늘 로그인했다고 업데이트
        update_mask = active_mask & (rng.random(len(df)) < DAU_RATE)
        df.loc[update_mask, "last_login_date"] = YESTERDAY
        df.to_parquet(OUT_FILE, index=False)
        print(f"[users] 갱신 완료 ({update_mask.sum()}명 last_login_date 업데이트)")

    return df


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "backfill"
    run(mode)
