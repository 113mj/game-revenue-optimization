"""
DAG 2 - 매주 월요일 새벽 3시 자동 실행

Task 흐름:
  1. rebuild_analytics     : analytics 테이블 재계산 (events / purchases / marketing_events)
  2. rebuild_snapshots     : user_daily_snapshot / daily_game_metrics 재계산
  3. upload_to_bigquery    : BigQuery 업로드 (WRITE_TRUNCATE)
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/data_generator")

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "game-revenue",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


# ── Task 1: analytics 테이블 재계산 ───────────────────────────────────────────
def rebuild_analytics(**context) -> None:
    import generate_analytics
    log.info("DAG2 - analytics 테이블 재계산 시작")
    generate_analytics.run("backfill")
    log.info("DAG2 - analytics 테이블 재계산 완료")


# ── Task 2: snapshot 재계산 ───────────────────────────────────────────────────
def rebuild_snapshots(**context) -> None:
    import generate_snapshots
    log.info("DAG2 - snapshots 재계산 시작")
    generate_snapshots.run("backfill")
    log.info("DAG2 - snapshots 재계산 완료")


# ── Task 3: BigQuery 업로드 ───────────────────────────────────────────────────
def upload_to_bigquery(**context) -> None:
    import pandas as pd
    from google.cloud import bigquery
    from config import GCP_PROJECT_ID, BQ_DATASET, OUTPUT_DIR

    RAW_DIR  = OUTPUT_DIR / "raw"
    ANAL_DIR = OUTPUT_DIR / "analytics"

    bq = bigquery.Client(project=GCP_PROJECT_ID)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )

    upload_targets = [
        # (테이블명, 로컬 경로)
        ("events",              ANAL_DIR / "events"),
        ("purchases",           ANAL_DIR / "purchases"),
        ("marketing_events",    ANAL_DIR / "marketing_events"),
        ("user_daily_snapshot", RAW_DIR  / "user_daily_snapshot" / "user_daily_snapshot.parquet"),
        ("daily_game_metrics",  RAW_DIR  / "daily_game_metrics"  / "daily_game_metrics.parquet"),
    ]

    for table_name, path in upload_targets:
        if not path.exists():
            log.warning(f"[{table_name}] 로컬 데이터 없음, skip")
            continue

        log.info(f"[{table_name}] BigQuery 업로드 중...")
        df = pd.read_parquet(path)
        table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{table_name}"
        job = bq.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()

        bq_table = bq.get_table(table_ref)
        log.info(f"[{table_name}] 완료 → {bq_table.num_rows:,}행")


# ── DAG 정의 ──────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag2_bq_aggregation",
    description="주간 analytics 재계산 → BigQuery 업로드",
    schedule="0 3 * * 1",          # 매주 월요일 새벽 3시
    start_date=datetime(2025, 7, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["weekly", "bigquery", "aggregation"],
) as dag:

    t1 = PythonOperator(
        task_id="rebuild_analytics",
        python_callable=rebuild_analytics,
    )

    t2 = PythonOperator(
        task_id="rebuild_snapshots",
        python_callable=rebuild_snapshots,
    )

    t3 = PythonOperator(
        task_id="upload_to_bigquery",
        python_callable=upload_to_bigquery,
    )

    [t1, t2] >> t3
