"""
DAG 1 - 매일 새벽 2시 자동 실행

Task 흐름:
  1. generate_incremental  : 어제 날짜 데이터 생성 (16개 테이블)
  2. upload_raw_to_gcs     : raw 데이터 → GCS 업로드
  3. notify_done           : 완료 로그
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/data_generator")

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "game-revenue",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


# ── Task 1: 어제 데이터 생성 ──────────────────────────────────────────────────
def generate_incremental(**context) -> None:
    import main as pipeline
    log.info("DAG1 - incremental 데이터 생성 시작")
    pipeline.run("incremental")
    log.info("DAG1 - incremental 데이터 생성 완료")


# ── Task 2: GCS 업로드 ────────────────────────────────────────────────────────
def upload_to_gcs(**context) -> None:
    import time
    from google.cloud import storage
    from config import GCS_BUCKET_NAME, GCP_PROJECT_ID, OUTPUT_DIR

    RAW_DIR = OUTPUT_DIR / "raw"
    GCS_TABLES = [
        "users", "unit_stats", "pilot_stats", "pilot_unit_synergy",
        "gacha_history", "gacha_event_log", "stage_clear_log",
        "stage_unit_log", "party_composition_log", "session_log",
        "user_unit_inventory", "user_daily_snapshot", "daily_game_metrics",
    ]

    log.info(f"DAG1 - GCS 업로드 시작 → gs://{GCS_BUCKET_NAME}/raw/")
    gcs = storage.Client(project=GCP_PROJECT_ID)
    bucket = gcs.bucket(GCS_BUCKET_NAME)

    # 어제 날짜 파티션만 업로드 (전체 재업로드 방지)
    yesterday = (context["data_interval_end"] - timedelta(days=1)).strftime("%Y-%m-%d")
    total = 0

    for table in GCS_TABLES:
        table_dir = RAW_DIR / table

        # 날짜 파티션 테이블: 어제 파티션만
        partition_dir = table_dir / f"date={yesterday}"
        if partition_dir.exists():
            for f in partition_dir.rglob("*.parquet"):
                blob_path = str(f.relative_to(OUTPUT_DIR))
                bucket.blob(blob_path).upload_from_filename(str(f))
                total += 1
        # 단일 파일 테이블 (users, unit_stats 등): 항상 최신 버전으로 덮어씀
        elif (table_dir / f"{table}.parquet").exists():
            f = table_dir / f"{table}.parquet"
            blob_path = str(f.relative_to(OUTPUT_DIR))
            bucket.blob(blob_path).upload_from_filename(str(f))
            total += 1

    log.info(f"DAG1 - GCS 업로드 완료: {total}개 파일 ({yesterday})")


# ── DAG 정의 ──────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag1_daily_data_pipeline",
    description="매일 데이터 생성 → GCS 업로드",
    schedule="0 2 * * *",          # 매일 새벽 2시
    start_date=datetime(2025, 7, 1),
    catchup=False,                  # 과거 날짜 자동 실행 방지
    default_args=DEFAULT_ARGS,
    tags=["daily", "data-generation", "gcs"],
) as dag:

    t1 = PythonOperator(
        task_id="generate_incremental",
        python_callable=generate_incremental,
    )

    t2 = PythonOperator(
        task_id="upload_raw_to_gcs",
        python_callable=upload_to_gcs,
    )

    t1 >> t2
