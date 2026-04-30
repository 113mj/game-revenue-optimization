"""
분석용 테이블 생성 (BigQuery 적재 대상)

 - 테이블 14: events           (Recalculate - user×date 집계)
 - 테이블 15: purchases        (Append + Recalculate - 결제 로그)
 - 테이블 16: marketing_events (Append - CTR/CVR/A-B 테스트)

purchases는 gacha/session 로그와 독립적으로 유저 수준에서 직접 생성.
marketing_events는 A/B 그룹별 CVR 차이를 명시적으로 반영.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import (
    START_DATE, YESTERDAY, RANDOM_SEED, OUTPUT_DIR,
    PAYING_USER_RATE, CVR_A, CVR_B, CTR_BASE,
    ITEM_CATEGORIES, ITEM_CATEGORY_WEIGHTS,
    PURCHASE_AMOUNTS, PURCHASE_WEIGHTS,
)

RAW_DIR  = OUTPUT_DIR / "raw"
ANAL_DIR = OUTPUT_DIR / "analytics"


def date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def partition_path(base: Path, table: str, d: date) -> Path:
    return base / table / f"date={d.isoformat()}" / "data.parquet"


# ── events (분석용) ───────────────────────────────────────────────────────────
def gen_events_day(
    d: date,
    users_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """session_log + stage_clear_log + gacha_history를 user×date로 집계.
    직접 집계 대신 근사 생성 (원본 파티션 파일 크기 고려).
    """
    # 활성 유저
    active = users_df[
        (users_df["signup_date"] <= d) & (users_df["last_login_date"] >= d)
    ]
    dau_rate = np.clip(rng.normal(0.30, 0.05), 0.05, 0.90)
    day_users = active[rng.random(len(active)) < dau_rate].copy()
    if day_users.empty:
        return pd.DataFrame()

    n = len(day_users)

    # 연속 로그인일수: 단순 랜덤 (실제라면 이전 날짜 누적)
    consec_login = rng.integers(1, 60, n)

    df = pd.DataFrame({
        "user_id":               day_users["user_id"].values,
        "event_date":            d,
        "login_yn":              True,
        "playtime_minutes":      np.clip(rng.lognormal(3.2, 0.8, n).astype(int), 1, 300),
        "level":                 np.clip(rng.integers(1, 201, n), 1, 200),
        "stage_clear_yn":        rng.random(n) < 0.65,
        "gacha_result":          rng.choice(["N","R","SR","SSR"], n, p=[0.55,0.27,0.13,0.05]),
        "gacha_try_count":       rng.poisson(0.7, n).clip(0, 20),
        "item_use_count":        rng.poisson(0.3, n).clip(0, 10),
        "session_count":         rng.poisson(1.5, n).clip(1, 8),
        "consecutive_login_days": consec_login,
    })
    return df


# ── purchases (분석용) ────────────────────────────────────────────────────────
def gen_purchases_day(
    d: date,
    users_df: pd.DataFrame,
    paying_user_ids: set[str],
    rng: np.random.Generator,
    purchase_counts: dict[str, int],
) -> pd.DataFrame:
    """결제 로그 생성. paying_user_ids 중 일부가 당일 결제."""
    active = users_df[
        (users_df["signup_date"] <= d) & (users_df["last_login_date"] >= d)
        & users_df["user_id"].isin(paying_user_ids)
    ]
    if active.empty:
        return pd.DataFrame()

    # 결제 유저 중 당일 결제 확률: ~5%
    purchasers = active[rng.random(len(active)) < 0.05]
    if purchasers.empty:
        return pd.DataFrame()

    rows = []
    for uid in purchasers["user_id"].values:
        n_purchases = int(rng.poisson(1.3))
        if n_purchases == 0:
            continue
        for _ in range(n_purchases):
            purchase_counts[uid] = purchase_counts.get(uid, 0) + 1
            amount = rng.choice(PURCHASE_AMOUNTS, p=PURCHASE_WEIGHTS)
            rows.append({
                "user_id":          uid,
                "purchase_date":    d,
                "purchase_amount":  float(amount),
                "item_category":    rng.choice(ITEM_CATEGORIES, p=ITEM_CATEGORY_WEIGHTS),
                "is_first_purchase": purchase_counts[uid] == 1,
                "payment_count":    purchase_counts[uid],
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # cumulative_amount: 이 파티션 내에서만 계산 (전체 집계는 snapshot에서)
    df["cumulative_amount"] = df.groupby("user_id")["purchase_amount"].cumsum()
    return df


# ── marketing_events (분석용) ─────────────────────────────────────────────────
def gen_marketing_events_day(
    d: date,
    users_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """노출→클릭→전환 퍼널. A/B 그룹별 CVR 차이 반영."""
    active = users_df[
        (users_df["signup_date"] <= d) & (users_df["last_login_date"] >= d)
    ]
    if active.empty:
        return pd.DataFrame()

    n = len(active)
    # 캠페인 노출: 활성 유저의 40%에게 노출
    impression_mask = rng.random(n) < 0.40
    impressed = active[impression_mask].copy()
    if impressed.empty:
        return pd.DataFrame()

    n_imp = len(impressed)
    ab_groups = impressed["ab_group"].values

    # CTR: 클릭률 15% + 노이즈
    ctr = np.where(ab_groups == "B", CTR_BASE + 0.02, CTR_BASE)
    click_yn = rng.random(n_imp) < ctr

    # CVR: 클릭 유저 중 전환 (A=3%, B=4.5%)
    cvr = np.where(ab_groups == "B", CVR_B, CVR_A)
    conversion_yn = click_yn & (rng.random(n_imp) < cvr)

    # 보상 금액 (A: 1000, B: 1500)
    reward = np.where(ab_groups == "B", 1500.0, 1000.0)

    # 광고 비용/수익
    ad_cost    = np.where(impression_mask[impression_mask], rng.uniform(0.01, 0.05, n_imp), 0.0)
    ad_revenue = np.where(conversion_yn, rng.uniform(500, 5000, n_imp), 0.0)

    return pd.DataFrame({
        "user_id":        impressed["user_id"].values,
        "campaign_date":  d,
        "impression_yn":  True,
        "click_yn":       click_yn,
        "conversion_yn":  conversion_yn,
        "reward_amount":  reward,
        "ad_cost":        np.round(ad_cost, 4),
        "ad_revenue":     np.round(ad_revenue, 2),
    })


# ── 실행 ──────────────────────────────────────────────────────────────────────
def run(mode: str = "backfill", users_df: pd.DataFrame | None = None) -> None:
    if users_df is None:
        users_df = pd.read_parquet(RAW_DIR / "users" / "users.parquet")

    rng_base = RANDOM_SEED + 9999

    # paying users: 전체 유저의 10%
    rng_init = np.random.default_rng(rng_base)
    all_user_ids = users_df["user_id"].values
    paying_user_ids: set[str] = set(
        rng_init.choice(all_user_ids, size=int(len(all_user_ids) * PAYING_USER_RATE), replace=False)
    )
    purchase_counts: dict[str, int] = {}

    if mode == "backfill":
        dates = list(date_range(START_DATE, YESTERDAY))
        print(f"[analytics] backfill {len(dates)}일치 생성 시작...")
        for i, d in enumerate(dates):
            rng = np.random.default_rng(rng_base + i)
            _save_day(d, users_df, paying_user_ids, purchase_counts, rng)
            if (i + 1) % 30 == 0:
                print(f"  → {i+1}/{len(dates)}일 완료 ({d})")
        print("[analytics] backfill 완료")

    else:  # incremental
        d = YESTERDAY
        day_idx = (d - START_DATE).days
        rng = np.random.default_rng(rng_base + day_idx)
        _save_day(d, users_df, paying_user_ids, purchase_counts, rng, overwrite=True)
        print(f"[analytics] incremental 완료 ({d})")


def _save_day(
    d: date,
    users_df: pd.DataFrame,
    paying_user_ids: set[str],
    purchase_counts: dict[str, int],
    rng: np.random.Generator,
    overwrite: bool = False,
) -> None:
    for table, gen_fn in [
        ("events",           lambda: gen_events_day(d, users_df, rng)),
        ("purchases",        lambda: gen_purchases_day(d, users_df, paying_user_ids, rng, purchase_counts)),
        ("marketing_events", lambda: gen_marketing_events_day(d, users_df, rng)),
    ]:
        path = partition_path(ANAL_DIR, table, d)
        if path.exists() and not overwrite:
            continue
        df = gen_fn()
        if df is not None and not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "backfill")
