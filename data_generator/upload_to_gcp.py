"""
로컬 parquet → GCS (raw 테이블) + BigQuery (analytics 테이블) 업로드

사용법:
    python upload_to_gcp.py           # 전체 업로드
    python upload_to_gcp.py --gcs     # GCS만
    python upload_to_gcp.py --bq      # BigQuery만
    python upload_to_gcp.py --check   # GCP 연결 테스트만
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from config import GCP_PROJECT_ID, GCS_BUCKET_NAME, BQ_DATASET, OUTPUT_DIR

RAW_DIR  = OUTPUT_DIR / "raw"
ANAL_DIR = OUTPUT_DIR / "analytics"

# BigQuery에 올릴 analytics 테이블 목록
BQ_TABLES = ["events", "purchases", "marketing_events"]

# GCS에 올릴 raw 테이블 목록
GCS_RAW_TABLES = [
    "users",
    "unit_stats",
    "pilot_stats",
    "pilot_unit_synergy",
    "gacha_history",
    "gacha_event_log",
    "stage_clear_log",
    "stage_unit_log",
    "party_composition_log",
    "session_log",
    "user_unit_inventory",
    "user_daily_snapshot",
    "daily_game_metrics",
]


# ── GCP 연결 테스트 ────────────────────────────────────────────────────────────
def check_connection() -> bool:
    print("GCP 연결 테스트 중...")
    try:
        from google.cloud import storage, bigquery

        # GCS 버킷 접근 확인
        gcs = storage.Client(project=GCP_PROJECT_ID)
        bucket = gcs.bucket(GCS_BUCKET_NAME)
        if not bucket.exists():
            print(f"  [경고] GCS 버킷 '{GCS_BUCKET_NAME}' 이 존재하지 않습니다.")
            print(f"  GCP 콘솔에서 버킷을 먼저 생성해주세요.")
            return False
        print(f"  [OK] GCS 버킷 '{GCS_BUCKET_NAME}' 접근 성공")

        # BigQuery 데이터셋 접근 확인
        bq = bigquery.Client(project=GCP_PROJECT_ID)
        dataset_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}"
        try:
            bq.get_dataset(dataset_ref)
            print(f"  [OK] BigQuery 데이터셋 '{BQ_DATASET}' 접근 성공")
        except Exception:
            print(f"  [경고] BigQuery 데이터셋 '{BQ_DATASET}' 없음 → 자동 생성합니다.")
            bq.create_dataset(dataset_ref, exists_ok=True)
            print(f"  [OK] BigQuery 데이터셋 '{BQ_DATASET}' 생성 완료")

        print("GCP 연결 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"  [ERROR] GCP 연결 실패: {e}")
        print("\n  체크리스트:")
        print("  1. .env 파일의 GOOGLE_APPLICATION_CREDENTIALS 경로가 맞는지 확인")
        print("  2. 서비스 계정에 Storage Admin + BigQuery Admin 권한이 있는지 확인")
        print("  3. pip install google-cloud-storage google-cloud-bigquery 설치 여부 확인")
        return False


# ── GCS 업로드 ─────────────────────────────────────────────────────────────────
def upload_to_gcs() -> None:
    from google.cloud import storage

    print(f"\n[GCS] 업로드 시작 → gs://{GCS_BUCKET_NAME}/raw/")
    gcs = storage.Client(project=GCP_PROJECT_ID)
    bucket = gcs.bucket(GCS_BUCKET_NAME)

    total_files = 0
    start = time.time()

    for table in GCS_RAW_TABLES:
        table_dir = RAW_DIR / table
        if not table_dir.exists():
            print(f"  [SKIP] {table} (로컬 데이터 없음)")
            continue

        parquet_files = sorted(table_dir.rglob("*.parquet"))
        if not parquet_files:
            print(f"  [SKIP] {table} (parquet 없음)")
            continue

        for local_path in parquet_files:
            # GCS 경로: raw/{table}/date=YYYY-MM-DD/data.parquet
            relative = local_path.relative_to(OUTPUT_DIR)
            gcs_path = str(relative)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            total_files += 1

        print(f"  [OK] {table} ({len(parquet_files)}개 파일)")

    elapsed = time.time() - start
    print(f"[GCS] 완료: 총 {total_files}개 파일 ({elapsed:.1f}초)\n")


# ── BigQuery 업로드 ────────────────────────────────────────────────────────────
def upload_to_bq() -> None:
    import pandas as pd
    from google.cloud import bigquery

    print(f"\n[BigQuery] 업로드 시작 → {GCP_PROJECT_ID}.{BQ_DATASET}")
    bq = bigquery.Client(project=GCP_PROJECT_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # 덮어쓰기
        source_format=bigquery.SourceFormat.PARQUET,
        autodetect=True,
    )

    start = time.time()

    for table in BQ_TABLES:
        table_dir = ANAL_DIR / table
        if not table_dir.exists():
            print(f"  [SKIP] {table} (로컬 데이터 없음)")
            continue

        parquet_files = sorted(table_dir.rglob("*.parquet"))
        if not parquet_files:
            print(f"  [SKIP] {table} (parquet 없음)")
            continue

        # parquet 파티션 전체를 pandas로 읽어서 BQ에 업로드
        print(f"  [{table}] 로딩 중... ({len(parquet_files)}개 파티션)")
        df = pd.read_parquet(table_dir)

        table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{table}"
        job = bq.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # 완료 대기

        bq_table = bq.get_table(table_ref)
        print(f"  [OK] {table} → {bq_table.num_rows:,}행 업로드 완료")

    # user_daily_snapshot, daily_game_metrics도 BQ에 올림
    for table, parquet_path in [
        ("user_daily_snapshot", RAW_DIR / "user_daily_snapshot" / "user_daily_snapshot.parquet"),
        ("daily_game_metrics",  RAW_DIR / "daily_game_metrics"  / "daily_game_metrics.parquet"),
    ]:
        if not parquet_path.exists():
            print(f"  [SKIP] {table} (로컬 데이터 없음)")
            continue

        print(f"  [{table}] 업로드 중...")
        df = pd.read_parquet(parquet_path)
        table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{table}"
        job = bq.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        bq_table = bq.get_table(table_ref)
        print(f"  [OK] {table} → {bq_table.num_rows:,}행 업로드 완료")

    elapsed = time.time() - start
    print(f"[BigQuery] 완료 ({elapsed:.1f}초)\n")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="GCP 업로드 스크립트")
    parser.add_argument("--check", action="store_true", help="GCP 연결 테스트만 실행")
    parser.add_argument("--gcs",   action="store_true", help="GCS 업로드만 실행")
    parser.add_argument("--bq",    action="store_true", help="BigQuery 업로드만 실행")
    args = parser.parse_args()

    # 연결 테스트 항상 먼저
    if not check_connection():
        sys.exit(1)

    if args.check:
        return

    if args.gcs:
        upload_to_gcs()
    elif args.bq:
        upload_to_bq()
    else:
        # 기본: 둘 다 실행
        upload_to_gcs()
        upload_to_bq()

    print("모든 업로드 완료!")
    print(f"  GCS:       gs://{GCS_BUCKET_NAME}/raw/")
    print(f"  BigQuery:  {GCP_PROJECT_ID}.{BQ_DATASET}")


if __name__ == "__main__":
    main()
