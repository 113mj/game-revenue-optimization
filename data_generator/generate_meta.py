"""
정적/반정적 메타 테이블 생성
 - 테이블 2: unit_stats       (갱신: Append - 신규 유닛)
 - 테이블 3: pilot_stats      (갱신: Append - 신규 파일럿)
 - 테이블 4: pilot_unit_synergy (갱신: Append - 신규 시너지)
 - 테이블 6: gacha_event_log  (갱신: Append+Update - 픽업 이벤트)

backfill / incremental 모두 전체 재생성 (행 수가 적어 overhead 없음).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from config import (
    START_DATE, YESTERDAY, RANDOM_SEED, OUTPUT_DIR,
    N_UNITS, N_PILOTS, N_GACHA_EVENTS,
    GRADES, UNIT_TYPES, UNIT_SERIES,
    PILOT_SKILLS, SYNERGY_GRADES, SYNERGY_GRADE_WEIGHTS,
    STRONG_STAGE_TYPES,
)

RAW_DIR = OUTPUT_DIR / "raw"


# ── unit_stats ────────────────────────────────────────────────────────────────
def generate_unit_stats(rng: np.random.Generator) -> pd.DataFrame:
    grade_weights = [0.40, 0.30, 0.20, 0.10]   # N/R/SR/SSR 비율
    df = pd.DataFrame({
        "unit_id":          range(1, N_UNITS + 1),
        "unit_name":        [f"Unit_{i:03d}" for i in range(1, N_UNITS + 1)],
        "unit_grade":       rng.choice(GRADES, size=N_UNITS, p=grade_weights),
        "unit_type":        rng.choice(UNIT_TYPES, size=N_UNITS),
        "unit_series":      rng.choice(UNIT_SERIES, size=N_UNITS),
        "strong_stage_type": rng.choice(STRONG_STAGE_TYPES, size=N_UNITS),
    })
    return df


# ── pilot_stats ───────────────────────────────────────────────────────────────
def generate_pilot_stats(rng: np.random.Generator) -> pd.DataFrame:
    grade_weights = [0.40, 0.30, 0.20, 0.10]
    df = pd.DataFrame({
        "pilot_id":    range(1, N_PILOTS + 1),
        "pilot_name":  [f"Pilot_{i:03d}" for i in range(1, N_PILOTS + 1)],
        "pilot_grade": rng.choice(GRADES, size=N_PILOTS, p=grade_weights),
        "pilot_skill": rng.choice(PILOT_SKILLS, size=N_PILOTS),
    })
    return df


# ── pilot_unit_synergy ────────────────────────────────────────────────────────
def generate_pilot_unit_synergy(
    rng: np.random.Generator,
    unit_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    n_synergies: int = 500,
) -> pd.DataFrame:
    """랜덤으로 pilot-unit 조합 500쌍 생성. 중복 없음."""
    unit_ids  = unit_df["unit_id"].values
    pilot_ids = pilot_df["pilot_id"].values

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_synergies:
        p = int(rng.choice(pilot_ids))
        u = int(rng.choice(unit_ids))
        pairs.add((p, u))

    pilots_arr, units_arr = zip(*pairs)
    df = pd.DataFrame({
        "pilot_id":    list(pilots_arr),
        "unit_id":     list(units_arr),
        "synergy_grade": rng.choice(
            SYNERGY_GRADES, size=n_synergies, p=SYNERGY_GRADE_WEIGHTS
        ),
        "bonus_rate": np.round(rng.uniform(1.05, 1.50, n_synergies), 2),
    })
    return df


# ── gacha_event_log ───────────────────────────────────────────────────────────
def generate_gacha_event_log(
    rng: np.random.Generator,
    unit_df: pd.DataFrame,
) -> pd.DataFrame:
    """픽업 이벤트 50건. 14일 단위 간격으로 분산."""
    # SR/SSR 유닛만 픽업 대상
    pickup_units = unit_df[unit_df["unit_grade"].isin(["SR", "SSR"])]["unit_id"].values
    if len(pickup_units) < N_GACHA_EVENTS:
        pickup_units = unit_df["unit_id"].values  # fallback

    total_days = (YESTERDAY - START_DATE).days
    # 이벤트 시작일: 14일마다 하나씩, 약간 노이즈 추가
    starts = []
    current = START_DATE
    for _ in range(N_GACHA_EVENTS):
        if current >= YESTERDAY:
            break
        starts.append(current)
        current += timedelta(days=14 + int(rng.integers(-2, 3)))

    n = len(starts)
    end_dates    = [s + timedelta(days=13) for s in starts]
    unit_choices = rng.choice(pickup_units, size=n, replace=True)

    # 집계 컬럼: 이벤트가 끝난 건만 집계값 존재
    total_gacha  = []
    unique_users = []
    for s, e in zip(starts, end_dates):
        if e < YESTERDAY:   # 이벤트 종료됨
            total_gacha.append(int(rng.integers(5_000, 50_000)))
            unique_users.append(int(rng.integers(500, 8_000)))
        else:               # 진행 중 또는 미래
            total_gacha.append(int(rng.integers(100, total_days * 50)))
            unique_users.append(int(rng.integers(50, 2_000)))

    df = pd.DataFrame({
        "event_id":         range(1, n + 1),
        "unit_id":          unit_choices,
        "event_start_date": starts,
        "event_end_date":   end_dates,
        "total_gacha_count": total_gacha,
        "unique_user_count": unique_users,
    })
    return df


# ── 실행 ──────────────────────────────────────────────────────────────────────
def run(mode: str = "backfill") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    results: dict[str, pd.DataFrame] = {}

    tables = {
        "unit_stats":         lambda: generate_unit_stats(rng),
        "pilot_stats":        lambda: generate_pilot_stats(rng),
    }

    for name, gen_fn in tables.items():
        out = RAW_DIR / name / f"{name}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df = gen_fn()
        df.to_parquet(out, index=False)
        results[name] = df
        print(f"[{name}] 저장 완료 → {out}  ({len(df):,}행)")

    # synergy는 unit/pilot 의존
    syn_out = RAW_DIR / "pilot_unit_synergy" / "pilot_unit_synergy.parquet"
    syn_out.parent.mkdir(parents=True, exist_ok=True)
    syn_df = generate_pilot_unit_synergy(rng, results["unit_stats"], results["pilot_stats"])
    syn_df.to_parquet(syn_out, index=False)
    results["pilot_unit_synergy"] = syn_df
    print(f"[pilot_unit_synergy] 저장 완료 → {syn_out}  ({len(syn_df):,}행)")

    # gacha_event_log
    gel_out = RAW_DIR / "gacha_event_log" / "gacha_event_log.parquet"
    gel_out.parent.mkdir(parents=True, exist_ok=True)
    gel_df = generate_gacha_event_log(rng, results["unit_stats"])
    gel_df.to_parquet(gel_out, index=False)
    results["gacha_event_log"] = gel_df
    print(f"[gacha_event_log] 저장 완료 → {gel_out}  ({len(gel_df):,}행)")

    return results


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "backfill")
