"""
테이블 11: user_unit_inventory
갱신 유형: Append (신규 획득) + Update (강화 시 enhance_level 갱신)

gacha_history를 집계해 유저별 보유 유닛 목록을 구성한 뒤,
강화 레벨과 즐겨찾기 여부를 랜덤으로 부여.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, RANDOM_SEED

RAW_DIR = OUTPUT_DIR / "raw"
OUT_FILE = RAW_DIR / "user_unit_inventory" / "user_unit_inventory.parquet"


def build_inventory(
    gacha_dir: Path,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """gacha_history 파티션을 전부 읽어 user_unit_inventory 구성."""
    parquet_files = sorted(gacha_dir.glob("date=*/data.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"gacha_history 파티션을 찾을 수 없습니다: {gacha_dir}"
        )

    # 필요한 컬럼만 로드해 메모리 절약
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f, columns=["user_id", "gacha_date", "result_unit_id"])
        dfs.append(df)
    gacha = pd.concat(dfs, ignore_index=True)

    # 같은 유저가 같은 유닛을 여러 번 획득할 수 있음 → 첫 획득일만 사용
    gacha["gacha_date"] = pd.to_datetime(gacha["gacha_date"])
    first_acquired = (
        gacha.groupby(["user_id", "result_unit_id"])["gacha_date"]
        .min()
        .reset_index()
        .rename(columns={"result_unit_id": "unit_id", "gacha_date": "acquired_date"})
    )

    n = len(first_acquired)

    # 강화 레벨: 대부분 낮은 레벨 (0~10), 정규화해서 사용
    _p = np.array([0.30, 0.18, 0.14, 0.11, 0.09, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005])
    _p /= _p.sum()
    enhance_levels = rng.choice(range(11), size=n, p=_p)

    is_favorite = rng.random(n) < 0.15  # 15%를 즐겨찾기

    inventory = first_acquired.copy()
    inventory["enhance_level"] = enhance_levels
    inventory["is_favorite"]   = is_favorite
    inventory["acquired_date"] = inventory["acquired_date"].dt.date

    return inventory.reset_index(drop=True)


def run(mode: str = "backfill") -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1000)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    gacha_dir = RAW_DIR / "gacha_history"

    if mode == "incremental" and OUT_FILE.exists():
        # incremental: 어제 가챠 결과만 추가
        print("[user_unit_inventory] incremental: 어제 신규 획득 유닛 추가 중...")
        existing = pd.read_parquet(OUT_FILE)

        from config import YESTERDAY
        yesterday_file = gacha_dir / f"date={YESTERDAY.isoformat()}" / "data.parquet"
        if not yesterday_file.exists():
            print("[user_unit_inventory] 어제 gacha_history 없음, skip")
            return existing

        new_gacha = pd.read_parquet(
            yesterday_file, columns=["user_id", "gacha_date", "result_unit_id"]
        )
        new_gacha = new_gacha.rename(columns={"result_unit_id": "unit_id"})
        new_gacha["acquired_date"] = pd.to_datetime(new_gacha["gacha_date"]).dt.date

        # 기존 보유 여부 확인
        existing_keys = set(zip(existing["user_id"], existing["unit_id"]))
        new_rows = new_gacha[
            ~new_gacha.apply(
                lambda r: (r["user_id"], r["unit_id"]) in existing_keys, axis=1
            )
        ][["user_id", "unit_id", "acquired_date"]].drop_duplicates()

        if not new_rows.empty:
            n = len(new_rows)
            _p = np.array([0.30, 0.18, 0.14, 0.11, 0.09, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005])
            _p /= _p.sum()
            new_rows = new_rows.copy()
            new_rows["enhance_level"] = rng.choice(range(11), size=n, p=_p)
            new_rows["is_favorite"]   = rng.random(n) < 0.15
            df = pd.concat([existing, new_rows], ignore_index=True)
        else:
            df = existing

        df.to_parquet(OUT_FILE, index=False)
        print(f"[user_unit_inventory] 갱신 완료 ({len(df):,}행)")

    else:
        print("[user_unit_inventory] backfill: gacha_history 전체 집계 중...")
        df = build_inventory(gacha_dir, rng)
        df.to_parquet(OUT_FILE, index=False)
        print(f"[user_unit_inventory] 저장 완료 → {OUT_FILE}  ({len(df):,}행)")

    return df


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "backfill")
