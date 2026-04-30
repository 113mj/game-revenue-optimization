"""
일별 Append 원천 테이블 생성 (날짜 파티션: data/raw/{table}/date=YYYY-MM-DD/data.parquet)

 - 테이블  5: gacha_history          (Append)
 - 테이블  7: stage_clear_log        (Append)
 - 테이블  8: stage_unit_log         (Append)
 - 테이블  9: party_composition_log  (Append)
 - 테이블 10: session_log            (Append)

backfill: START_DATE ~ YESTERDAY 전체 생성
incremental: YESTERDAY 1일치만 생성
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import (
    START_DATE, YESTERDAY, RANDOM_SEED, OUTPUT_DIR,
    N_STAGES, GRADES,
    GACHA_NORMAL_WEIGHTS, GACHA_PREMIUM_WEIGHTS, GACHA_EVENT_WEIGHTS,
    DAU_RATE, AVG_SESSION_MIN,
)

RAW_DIR = OUTPUT_DIR / "raw"
PARTY_SIZE = 5       # 파티 슬롯 수
GACHA_TYPES = ["normal", "premium", "event"]
GACHA_TYPE_WEIGHTS = [0.50, 0.30, 0.20]
SYNERGY_TYPES = ["공격형", "균형형", "방어형"]


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def partition_path(table: str, d: date) -> Path:
    return RAW_DIR / table / f"date={d.isoformat()}" / "data.parquet"


def active_users_on(
    users_df: pd.DataFrame,
    d: date,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """날짜 d에 로그인한 유저 샘플링."""
    eligible = users_df[
        (users_df["signup_date"] <= d) & (users_df["last_login_date"] >= d)
    ]
    # DAU = 활성 유저 × DAU_RATE + 노이즈
    dau_rate = min(1.0, max(0.05, rng.normal(DAU_RATE, 0.05)))
    mask = rng.random(len(eligible)) < dau_rate
    return eligible[mask].reset_index(drop=True)


# ── session_log ───────────────────────────────────────────────────────────────
def gen_session_log(
    day_users: pd.DataFrame,
    d: date,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if day_users.empty:
        return pd.DataFrame()

    n = len(day_users)
    # 유저당 평균 1.5 세션/일 (포아송)
    sessions_per_user = rng.poisson(1.5, n).clip(1, 6)
    user_ids = day_users["user_id"].repeat(sessions_per_user).values
    total_sessions = len(user_ids)

    # 로그인 시각: 하루 중 균등 분포 (피크: 오후 8~11시)
    hour_weights = np.array([
        1,1,1,1,1,1, 2,3,4,4,3,3,
        4,4,4,3,3,4, 5,6,8,8,6,3
    ], dtype=float)
    hour_weights /= hour_weights.sum()
    hours = rng.choice(np.arange(24), size=total_sessions, p=hour_weights)
    minutes = rng.integers(0, 60, total_sessions)
    login_times = [
        datetime(d.year, d.month, d.day, int(h), int(m))
        for h, m in zip(hours, minutes)
    ]

    durations = np.clip(
        rng.lognormal(mean=3.2, sigma=0.8, size=total_sessions).astype(int), 1, 240
    )
    logout_times = [
        lt + timedelta(minutes=int(dur))
        for lt, dur in zip(login_times, durations)
    ]

    session_ids = [f"S{d.strftime('%Y%m%d')}{i:07d}" for i in range(total_sessions)]

    return pd.DataFrame({
        "session_id":       session_ids,
        "user_id":          user_ids,
        "login_time":       login_times,
        "logout_time":      logout_times,
        "session_duration": durations,
    })


# ── gacha_history ─────────────────────────────────────────────────────────────
def gen_gacha_history(
    day_users: pd.DataFrame,
    d: date,
    rng: np.random.Generator,
    unit_ids: np.ndarray,
    event_units: np.ndarray | None = None,
) -> pd.DataFrame:
    if day_users.empty:
        return pd.DataFrame()

    # 가챠 참여 유저: DAU의 약 20%
    n_users = len(day_users)
    gacha_mask = rng.random(n_users) < 0.20
    gacha_users = day_users[gacha_mask]
    if gacha_users.empty:
        return pd.DataFrame()

    n = len(gacha_users)
    attempts_per_user = rng.poisson(3.5, n).clip(1, 30)
    user_ids_rep = gacha_users["user_id"].repeat(attempts_per_user).values
    total = len(user_ids_rep)

    gacha_types = rng.choice(GACHA_TYPES, size=total, p=GACHA_TYPE_WEIGHTS)

    result_grades = []
    for gt in gacha_types:
        if gt == "normal":
            result_grades.append(rng.choice(GRADES, p=GACHA_NORMAL_WEIGHTS))
        elif gt == "premium":
            result_grades.append(rng.choice(GRADES, p=GACHA_PREMIUM_WEIGHTS))
        else:
            result_grades.append(rng.choice(GRADES, p=GACHA_EVENT_WEIGHTS))

    # 이벤트 가챠면 픽업 유닛 가중치 증가
    result_unit_ids = []
    for gt in gacha_types:
        if gt == "event" and event_units is not None and len(event_units) > 0:
            pool = np.concatenate([event_units, rng.choice(unit_ids, 10)])
            result_unit_ids.append(int(rng.choice(pool)))
        else:
            result_unit_ids.append(int(rng.choice(unit_ids)))

    cost_map = {"normal": 100, "premium": 300, "event": 200}
    costs = [cost_map[gt] for gt in gacha_types]

    return pd.DataFrame({
        "user_id":        user_ids_rep,
        "gacha_date":     d,
        "gacha_type":     gacha_types,
        "result_grade":   result_grades,
        "result_unit_id": result_unit_ids,
        "cost_amount":    costs,
    })


# ── stage_clear_log ───────────────────────────────────────────────────────────
def gen_stage_clear_log(
    day_users: pd.DataFrame,
    d: date,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if day_users.empty:
        return pd.DataFrame()

    n = len(day_users)
    # 유저당 평균 5 스테이지 시도
    attempts_per_user = rng.poisson(5, n).clip(1, 20)
    user_ids_rep = day_users["user_id"].repeat(attempts_per_user).values
    total = len(user_ids_rep)

    stage_ids = rng.integers(1, N_STAGES + 1, total)
    attempt_counts = rng.integers(1, 6, total)
    # 클리어율: 스테이지가 높을수록 낮아짐
    clear_prob = np.clip(0.85 - (stage_ids / N_STAGES) * 0.50, 0.30, 0.85)
    clear_yn = rng.random(total) < clear_prob
    clear_times = np.where(
        clear_yn,
        rng.integers(60, 600, total),
        0,
    )
    item_use = rng.random(total) < 0.15

    return pd.DataFrame({
        "user_id":        user_ids_rep,
        "stage_id":       stage_ids,
        "attempt_count":  attempt_counts,
        "clear_yn":       clear_yn,
        "clear_time_sec": clear_times,
        "used_item_yn":   item_use,
        "event_date":     d,
    })


# ── stage_unit_log ────────────────────────────────────────────────────────────
def gen_stage_unit_log(
    stage_clear_df: pd.DataFrame,
    d: date,
    rng: np.random.Generator,
    unit_ids: np.ndarray,
    pilot_ids: np.ndarray,
) -> pd.DataFrame:
    if stage_clear_df.empty:
        return pd.DataFrame()

    n_battles = len(stage_clear_df)
    # 각 전투에 슬롯 1~5 유닛 투입
    slots_per_battle = rng.integers(1, PARTY_SIZE + 1, n_battles)
    battle_user_ids = stage_clear_df["user_id"].repeat(slots_per_battle).values
    battle_stage_ids = stage_clear_df["stage_id"].repeat(slots_per_battle).values
    battle_clear_yn  = stage_clear_df["clear_yn"].repeat(slots_per_battle).values
    total = len(battle_user_ids)

    slot_positions = []
    for slots in slots_per_battle:
        slot_positions.extend(range(1, int(slots) + 1))

    return pd.DataFrame({
        "user_id":       battle_user_ids,
        "stage_id":      battle_stage_ids,
        "unit_id":       rng.choice(unit_ids, total),
        "pilot_id":      rng.choice(pilot_ids, total),
        "slot_position": slot_positions,
        "clear_yn":      battle_clear_yn,
        "event_date":    d,
    })


# ── party_composition_log ─────────────────────────────────────────────────────
def gen_party_composition_log(
    stage_clear_df: pd.DataFrame,
    d: date,
    rng: np.random.Generator,
    unit_ids: np.ndarray,
) -> pd.DataFrame:
    if stage_clear_df.empty:
        return pd.DataFrame()

    n = len(stage_clear_df)
    rows = []
    for _, row in stage_clear_df.iterrows():
        chosen = rng.choice(unit_ids, PARTY_SIZE, replace=False)
        rows.append({
            "user_id":      row["user_id"],
            "stage_id":     row["stage_id"],
            "unit_ids":     ",".join(map(str, chosen)),
            "synergy_type": rng.choice(SYNERGY_TYPES),
            "clear_yn":     row["clear_yn"],
            "event_date":   d,
        })
    return pd.DataFrame(rows)


# ── 1일치 생성 ────────────────────────────────────────────────────────────────
def generate_one_day(
    d: date,
    users_df: pd.DataFrame,
    unit_ids: np.ndarray,
    pilot_ids: np.ndarray,
    event_units: np.ndarray | None,
    rng: np.random.Generator,
    overwrite: bool = False,
) -> None:
    # 이미 존재하면 skip (backfill 재실행 보호)
    session_part = partition_path("session_log", d)
    if session_part.exists() and not overwrite:
        return

    day_users = active_users_on(users_df, d, rng)

    # 1. session_log
    sess_df = gen_session_log(day_users, d, rng)
    _save(sess_df, "session_log", d)

    # 2. gacha_history
    gacha_df = gen_gacha_history(day_users, d, rng, unit_ids, event_units)
    _save(gacha_df, "gacha_history", d)

    # 3. stage_clear_log
    scl_df = gen_stage_clear_log(day_users, d, rng)
    _save(scl_df, "stage_clear_log", d)

    # 4. stage_unit_log (stage_clear_log 의존)
    if not scl_df.empty:
        sul_df = gen_stage_unit_log(scl_df, d, rng, unit_ids, pilot_ids)
        _save(sul_df, "stage_unit_log", d)

    # 5. party_composition_log (stage_clear_log 의존)
    if not scl_df.empty:
        # 성능상 stage당 1 파티 기록 (전체 로그 크기 관리)
        pcl_df = gen_party_composition_log(scl_df, d, rng, unit_ids)
        _save(pcl_df, "party_composition_log", d)


def _save(df: pd.DataFrame, table: str, d: date) -> None:
    if df.empty:
        return
    path = partition_path(table, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# ── 실행 ──────────────────────────────────────────────────────────────────────
def run(
    mode: str = "backfill",
    users_df: pd.DataFrame | None = None,
    meta: dict | None = None,
) -> None:
    from generate_users import run as run_users
    from generate_meta import run as run_meta

    if users_df is None:
        users_df = pd.read_parquet(OUTPUT_DIR / "raw" / "users" / "users.parquet")
    if meta is None:
        meta = {}

    unit_ids  = pd.read_parquet(RAW_DIR / "unit_stats"  / "unit_stats.parquet")["unit_id"].values
    pilot_ids = pd.read_parquet(RAW_DIR / "pilot_stats" / "pilot_stats.parquet")["pilot_id"].values

    # 진행 중 픽업 이벤트 유닛 파악
    gel_path = RAW_DIR / "gacha_event_log" / "gacha_event_log.parquet"
    gel_df = pd.read_parquet(gel_path) if gel_path.exists() else pd.DataFrame()

    # 시드 고정 (날짜별 다른 시드로 재현성 확보)
    base_seed = RANDOM_SEED

    if mode == "backfill":
        dates = list(date_range(START_DATE, YESTERDAY))
        print(f"[daily_raw] backfill {len(dates)}일치 생성 시작...")
        for i, d in enumerate(dates):
            rng = np.random.default_rng(base_seed + i)

            # 해당 날짜의 픽업 이벤트 유닛
            event_units = None
            if not gel_df.empty:
                active_events = gel_df[
                    (gel_df["event_start_date"] <= d) &
                    (gel_df["event_end_date"]   >= d)
                ]
                if not active_events.empty:
                    event_units = active_events["unit_id"].values

            generate_one_day(d, users_df, unit_ids, pilot_ids, event_units, rng)
            if (i + 1) % 30 == 0:
                print(f"  → {i+1}/{len(dates)}일 완료 ({d})")
        print("[daily_raw] backfill 완료")

    else:  # incremental
        d = YESTERDAY
        day_idx = (d - START_DATE).days
        rng = np.random.default_rng(base_seed + day_idx)

        event_units = None
        if not gel_df.empty:
            active_events = gel_df[
                (gel_df["event_start_date"] <= d) &
                (gel_df["event_end_date"]   >= d)
            ]
            if not active_events.empty:
                event_units = active_events["unit_id"].values

        generate_one_day(d, users_df, unit_ids, pilot_ids, event_units, rng, overwrite=True)
        print(f"[daily_raw] incremental 완료 ({d})")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "backfill")
