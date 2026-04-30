"""
DAG 3 - 매주 월요일 새벽 4시 자동 실행 (DAG 2 완료 후)

Task 흐름:
  1. load_features         : BigQuery에서 피처 로드
  2. train_churn_model     : XGBoost + LightGBM 이탈 예측 → MLflow 비교 로깅
  3. train_ltv_model       : XGBoost + LightGBM LTV 예측 → MLflow 비교 로깅
  4. update_churn_scores   : 최고 성능 모델로 이탈 스코어 갱신 → BigQuery 업데이트
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/data_generator")

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

DEFAULT_ARGS = {
    "owner": "game-revenue",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

CHURN_FEATURES = [
    "max_level", "total_playtime", "total_gacha", "total_spent",
    "active_days", "avg_daily_playtime", "avg_sessions", "max_consec_login",
]

LTV_FEATURES = [
    "max_level", "total_playtime", "total_gacha",
    "purchase_count", "avg_purchase_amount",
]


# ── Task 1: BigQuery에서 피처 로드 ─────────────────────────────────────────────
def load_features(**context) -> None:
    import pandas as pd
    from google.cloud import bigquery
    from config import GCP_PROJECT_ID, BQ_DATASET, OUTPUT_DIR

    log.info("DAG3 - BigQuery에서 피처 로드 중...")
    bq = bigquery.Client(project=GCP_PROJECT_ID)

    churn_query = f"""
    SELECT
        s.user_id,
        MAX(s.level)              AS max_level,
        MAX(s.total_playtime)     AS total_playtime,
        MAX(s.total_gacha)        AS total_gacha,
        MAX(s.total_spent)        AS total_spent,
        MAX(s.churn_risk_score)   AS churn_risk_score,
        COUNT(e.event_date)       AS active_days,
        AVG(e.playtime_minutes)   AS avg_daily_playtime,
        AVG(e.session_count)      AS avg_sessions,
        MAX(e.consecutive_login_days) AS max_consec_login,
        CASE
            WHEN MAX(s.last_login_date) < DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
            THEN 1 ELSE 0
        END AS is_churned
    FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.user_daily_snapshot` s
    LEFT JOIN `{GCP_PROJECT_ID}.{BQ_DATASET}.events` e
        ON s.user_id = e.user_id
    GROUP BY s.user_id
    """

    ltv_query = f"""
    SELECT
        s.user_id,
        MAX(s.level)            AS max_level,
        MAX(s.total_playtime)   AS total_playtime,
        MAX(s.total_gacha)      AS total_gacha,
        MAX(s.total_spent)      AS total_spent,
        COUNT(p.purchase_date)  AS purchase_count,
        AVG(p.purchase_amount)  AS avg_purchase_amount,
        COALESCE(MAX(p.cumulative_amount), 0) AS ltv_30d
    FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.user_daily_snapshot` s
    LEFT JOIN `{GCP_PROJECT_ID}.{BQ_DATASET}.purchases` p
        ON s.user_id = p.user_id
    GROUP BY s.user_id
    """

    churn_df = bq.query(churn_query).to_dataframe()
    ltv_df   = bq.query(ltv_query).to_dataframe()

    feat_dir = OUTPUT_DIR / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    churn_df.to_parquet(feat_dir / "churn_features.parquet", index=False)
    ltv_df.to_parquet(feat_dir   / "ltv_features.parquet",   index=False)

    log.info(f"DAG3 - 피처 로드 완료: churn={len(churn_df):,}행, ltv={len(ltv_df):,}행")


# ── Task 2: 이탈 예측 모델 학습 (XGBoost vs LightGBM) ─────────────────────────
def train_churn_model(**context) -> None:
    import mlflow
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score
    from xgboost import XGBClassifier
    from config import OUTPUT_DIR

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")

    df = pd.read_parquet(OUTPUT_DIR / "features" / "churn_features.parquet").dropna()
    X, y = df[CHURN_FEATURES], df["is_churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    today     = datetime.now().strftime("%Y%m%d")
    results   = {}
    scale_pos = float((y == 0).sum() / max((y == 1).sum(), 1))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"churn_xgb_{today}"):
        params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos, "random_state": 42, "eval_metric": "auc",
        }
        mlflow.log_params({"model": "xgboost", **params})

        xgb = XGBClassifier(**params)
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        f1  = f1_score(y_test, xgb.predict(X_test))
        mlflow.log_metrics({"auc": auc, "f1": f1, "train_size": len(X_train)})
        xgb.save_model(str(model_dir / "churn_xgb.json"))
        results["xgb"] = {"auc": auc, "f1": f1}
        log.info(f"XGBoost churn — AUC={auc:.4f}, F1={f1:.4f}")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"churn_lgbm_{today}"):
        lgbm_params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos, "random_state": 42, "verbose": -1,
        }
        mlflow.log_params({"model": "lightgbm", **lgbm_params})

        lgbm = lgb.LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
        f1  = f1_score(y_test, lgbm.predict(X_test))
        mlflow.log_metrics({"auc": auc, "f1": f1, "train_size": len(X_train)})
        lgbm.booster_.save_model(str(model_dir / "churn_lgbm.txt"))
        results["lgbm"] = {"auc": auc, "f1": f1}
        log.info(f"LightGBM churn — AUC={auc:.4f}, F1={f1:.4f}")

    # 최고 모델(AUC 기준) 메타 저장
    best = max(results, key=lambda k: results[k]["auc"])
    meta = {"best_churn_model": best, "churn_xgb": results["xgb"], "churn_lgbm": results["lgbm"]}
    with open(model_dir / "churn_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Best churn model: {best} (AUC={results[best]['auc']:.4f})")


# ── Task 3: LTV 예측 모델 학습 (XGBoost vs LightGBM) ─────────────────────────
def train_ltv_model(**context) -> None:
    import mlflow
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from xgboost import XGBRegressor
    from config import OUTPUT_DIR

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ltv_prediction")

    df = pd.read_parquet(OUTPUT_DIR / "features" / "ltv_features.parquet").dropna()
    df = df[df["ltv_30d"] > 0]
    X, y = df[LTV_FEATURES], df["ltv_30d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    today   = datetime.now().strftime("%Y%m%d")
    results = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"ltv_xgb_{today}"):
        params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
        }
        mlflow.log_params({"model": "xgboost", **params})

        xgb = XGBRegressor(**params)
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = xgb.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "train_size": len(X_train)})
        xgb.save_model(str(model_dir / "ltv_xgb.json"))
        results["xgb"] = {"rmse": rmse, "mae": mae}
        log.info(f"XGBoost LTV — RMSE={rmse:.2f}, MAE={mae:.2f}")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"ltv_lgbm_{today}"):
        lgbm_params = {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbose": -1,
        }
        mlflow.log_params({"model": "lightgbm", **lgbm_params})

        lgbm = lgb.LGBMRegressor(**lgbm_params)
        lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        y_pred = lgbm.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "train_size": len(X_train)})
        lgbm.booster_.save_model(str(model_dir / "ltv_lgbm.txt"))
        results["lgbm"] = {"rmse": rmse, "mae": mae}
        log.info(f"LightGBM LTV — RMSE={rmse:.2f}, MAE={mae:.2f}")

    # 최고 모델(RMSE 기준) 메타 저장
    best = min(results, key=lambda k: results[k]["rmse"])
    meta = {"best_ltv_model": best, "ltv_xgb": results["xgb"], "ltv_lgbm": results["lgbm"]}
    with open(model_dir / "ltv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Best LTV model: {best} (RMSE={results[best]['rmse']:.2f})")


# ── Task 4: 이탈 스코어 갱신 → BigQuery 업데이트 ──────────────────────────────
def update_churn_scores(**context) -> None:
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from xgboost import XGBClassifier
    from google.cloud import bigquery
    from config import GCP_PROJECT_ID, BQ_DATASET, OUTPUT_DIR

    model_dir = OUTPUT_DIR / "models"
    feat_path = OUTPUT_DIR / "features" / "churn_features.parquet"

    # 최고 모델 선택
    meta_path = model_dir / "churn_meta.json"
    best = "xgb"
    if meta_path.exists():
        with open(meta_path) as f:
            best = json.load(f).get("best_churn_model", "xgb")

    df = pd.read_parquet(feat_path).dropna(subset=CHURN_FEATURES)

    if best == "lgbm" and (model_dir / "churn_lgbm.txt").exists():
        booster = lgb.Booster(model_file=str(model_dir / "churn_lgbm.txt"))
        df["churn_risk_score"] = booster.predict(df[CHURN_FEATURES]).round(4)
        log.info("update_churn_scores: LightGBM 모델 사용")
    else:
        model_path = model_dir / "churn_xgb.json"
        if not model_path.exists():
            model_path = model_dir / "churn_model.json"
        if not model_path.exists():
            log.warning("DAG3 - churn 모델 없음, 스코어 갱신 skip")
            return
        model = XGBClassifier()
        model.load_model(str(model_path))
        df["churn_risk_score"] = model.predict_proba(df[CHURN_FEATURES])[:, 1].round(4)
        log.info("update_churn_scores: XGBoost 모델 사용")

    snap_path = OUTPUT_DIR / "raw" / "user_daily_snapshot" / "user_daily_snapshot.parquet"
    if snap_path.exists():
        snap = pd.read_parquet(snap_path)
        score_map = df.set_index("user_id")["churn_risk_score"].to_dict()
        snap["churn_risk_score"] = snap["user_id"].map(score_map).fillna(snap["churn_risk_score"])
        snap.to_parquet(snap_path, index=False)

        bq = bigquery.Client(project=GCP_PROJECT_ID)
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=True,
        )
        table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET}.user_daily_snapshot"
        bq.load_table_from_dataframe(snap, table_ref, job_config=job_config).result()

    log.info(f"DAG3 - 이탈 스코어 갱신 완료: {len(df):,}명 / best={best}")


# ── DAG 정의 ──────────────────────────────────────────────────────────────────
with DAG(
    dag_id="dag3_model_training",
    description="주간 이탈/LTV 모델 재학습 (XGBoost vs LightGBM) → MLflow 비교",
    schedule="0 4 * * 1",
    start_date=datetime(2025, 7, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["weekly", "ml", "mlflow"],
) as dag:

    t1 = PythonOperator(task_id="load_features",      python_callable=load_features)
    t2 = PythonOperator(task_id="train_churn_model",  python_callable=train_churn_model)
    t3 = PythonOperator(task_id="train_ltv_model",    python_callable=train_ltv_model)
    t4 = PythonOperator(task_id="update_churn_scores", python_callable=update_churn_scores)

    t1 >> [t2, t3] >> t4
