"""
Recalculate 테이블 생성 (전체 기간 재계산)

 - 테이블 12: user_daily_snapshot  (매일 전체 유저 상태 스냅샷)
 - 테이블 13: daily_game_metrics   (일별 게임 전체 지표)

두 테이블 모두 분석용 events / purchases 파티션을 집계해 생성.
backfill / incremental 구분 없이 항상 전체 재계산.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RANDOM_SEED, START_DATE, YESTERDAY

RAW_DIR  = OUTPUT_DIR / "raw"
ANAL_DIR = OUTPUT_DIR / "analytics"


def date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _load_partitions(base: Path, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(base.glob(f"{table}/date=*/data.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f, columns=columns) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ── user_daily_snapshot ───────────────────────────────────────────────────────
def generate_user_daily_snapshot() -> pd.DataFrame:
    """events + purchases를 집계해 유저별 일별 스냅샷 구성."""
    print("[user_daily_snapshot] events 로딩 중...")
    events = _load_partitions(
        ANAL_DIR, "events",
        ["user_id", "event_date", "playtime_minutes", "level", "gacha_try_count",
         "login_yn", "consecutive_login_days"],
    )
    print("[user_daily_snapshot] purchases 로딩 중...")
    purchases = _load_partitions(
        ANAL_DIR, "purchases",
        ["user_id", "purchase_date", "purchase_amount"],
    )

    if events.empty:
        print("[user_daily_snapshot] events 없음, skip")
        return pd.DataFrame()

    # 누적 결제 금액 (전체 기간 기준)
    if not purchases.empty:
        purchases["purchase_date"] = pd.to_datetime(purchases["purchase_date"]).dt.date
        cum_purchase = (
            purchases.sort_values("purchase_date")
            .groupby(["user_id", "purchase_date"])["purchase_amount"]
            .sum()
            .groupby(level=0).cumsum()
            .reset_index()
            .rename(columns={"purchase_amount": "total_spent", "purchase_date": "event_date"})
        )
    else:
        cum_purchase = pd.DataFrame(columns=["user_id", "event_date", "total_spent"])

    # 누적 플레이타임 / 가챠
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.date
    events = events.sort_values(["user_id", "event_date"])

    agg = events.groupby(["user_id", "event_date"]).agg(
        playtime_minutes=("playtime_minutes", "sum"),
        level=("level", "max"),
        gacha_try_count=("gacha_try_count", "sum"),
    ).reset_index()

    # 누적값 계산
    agg["total_playtime"] = agg.groupby("user_id")["playtime_minutes"].cumsum()
    agg["total_gacha"]    = agg.groupby("user_id")["gacha_try_count"].cumsum()

    snapshot = agg[["user_id", "event_date", "level", "total_playtime", "total_gacha"]].copy()
    snapshot = snapshot.rename(columns={"event_date": "snapshot_date"})

    # 결제 금액 병합
    if not cum_purchase.empty:
        cum_purchase["event_date"] = pd.to_datetime(cum_purchase["event_date"]).dt.date
        snapshot = snapshot.merge(
            cum_purchase.rename(columns={"event_date": "snapshot_date"}),
            on=["user_id", "snapshot_date"], how="left",
        )
    snapshot["total_spent"] = snapshot.get("total_spent", 0.0).fillna(0.0)

    # last_login_date: 해당 날짜 자체
    snapshot["last_login_date"] = snapshot["snapshot_date"]

    # churn_risk_score: 더미 스코어 (실제 모델 대체)
    rng = np.random.default_rng(RANDOM_SEED + 2000)
    snapshot["churn_risk_score"] = np.round(rng.beta(2, 5, len(snapshot)), 4)

    return snapshot[[
        "user_id", "snapshot_date", "level", "total_playtime",
        "total_gacha", "total_spent", "last_login_date", "churn_risk_score",
    ]]


# ── daily_game_metrics ────────────────────────────────────────────────────────
def generate_daily_game_metrics() -> pd.DataFrame:
    """events + purchases + session_log 집계로 게임 전체 KPI 산출."""
    print("[daily_game_metrics] 데이터 로딩 중...")

    events = _load_partitions(
        ANAL_DIR, "events",
        ["user_id", "event_date", "login_yn", "gacha_try_count"],
    )
    purchases = _load_partitions(
        ANAL_DIR, "purchases",
        ["user_id", "purchase_date", "purchase_amount"],
    )

    users = pd.read_parquet(RAW_DIR / "users" / "users.parquet",
                            columns=["user_id", "signup_date"])

    if events.empty:
        return pd.DataFrame()

    events["event_date"]    = pd.to_datetime(events["event_date"]).dt.date
    users["signup_date"]    = pd.to_datetime(users["signup_date"]).dt.date

    rows = []
    for d in date_range(START_DATE, YESTERDAY):
        day_events = events[events["event_date"] == d]
        dau = int(day_events["login_yn"].sum()) if not day_events.empty else 0
        dnu = int((users["signup_date"] == d).sum())

        # MAU: 해당 월 전체 DAU 근사 (30일 rolling)
        month_start = d.replace(day=1)
        month_events = events[
            (events["event_date"] >= month_start) & (events["event_date"] <= d)
        ]
        mau = int(month_events["user_id"].nunique()) if not month_events.empty else 0

        # 매출
        if not purchases.empty:
            purchases["purchase_date"] = pd.to_datetime(purchases["purchase_date"]).dt.date
            day_purchases = purchases[purchases["purchase_date"] == d]
            total_rev = float(day_purchases["purchase_amount"].sum())
            paying_users = day_purchases["user_id"].nunique()
        else:
            total_rev = 0.0
            paying_users = 0

        arpu  = round(total_rev / dau, 4) if dau > 0 else 0.0
        arppu = round(total_rev / paying_users, 2) if paying_users > 0 else 0.0
        pur   = round(paying_users / dau, 4) if dau > 0 else 0.0
        total_gacha = int(day_events["gacha_try_count"].sum()) if not day_events.empty else 0

        # PCU / ACU: session_log 없이 근사
        pcu = int(dau * 0.12)
        acu = round(dau * 0.06, 1)

        rows.append({
            "metric_date":       d,
            "DAU":               dau,
            "DNU":               dnu,
            "MAU":               mau,
            "PCU":               pcu,
            "ACU":               acu,
            "total_revenue":     round(total_rev, 2),
            "total_gacha_count": total_gacha,
            "PUR":               pur,
            "ARPU":              arpu,
            "ARPPU":             arppu,
        })

    return pd.DataFrame(rows)


# ── 실행 ──────────────────────────────────────────────────────────────────────
def run(mode: str = "backfill") -> None:
    # user_daily_snapshot
    snap_out = RAW_DIR / "user_daily_snapshot" / "user_daily_snapshot.parquet"
    snap_out.parent.mkdir(parents=True, exist_ok=True)
    snap_df = generate_user_daily_snapshot()
    if not snap_df.empty:
        snap_df.to_parquet(snap_out, index=False)
        print(f"[user_daily_snapshot] 저장 완료 → {snap_out}  ({len(snap_df):,}행)")

    # daily_game_metrics
    metrics_out = RAW_DIR / "daily_game_metrics" / "daily_game_metrics.parquet"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = generate_daily_game_metrics()
    if not metrics_df.empty:
        metrics_df.to_parquet(metrics_out, index=False)
        print(f"[daily_game_metrics] 저장 완료 → {metrics_out}  ({len(metrics_df):,}행)")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "backfill")
