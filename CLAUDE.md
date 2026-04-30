# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Game User Revenue Optimization System** — a portfolio project for Marketing Strategy DS / AD-Tech DS roles. It simulates a mobile turn-based RPG ("Chronicle Tactics") with 30,000 synthetic users over 6 months. The goal is to demonstrate end-to-end MLOps: data pipeline automation, funnel/cohort analysis, A/B testing with causal inference, churn/LTV prediction, and a Streamlit dashboard.

## Environment Setup

```bash
# GCP credentials and project config are in .env
source .env   # or export vars manually

# Start all services (Airflow, MLflow, Streamlit) via Docker Compose
docker-compose up -d
```

Key environment variables (see `.env`):
- `GOOGLE_APPLICATION_CREDENTIALS` — service account JSON path
- `GCP_PROJECT_ID=game-revenue-optimization`
- `GCS_BUCKET_NAME=game-revenue-raw-data-jmj`
- `BIGQUERY_DATASET=game_analytics`

## Architecture

```
[Python + Faker]  →  DAG 1 (daily)  →  GCS (raw partitioned data)
                                              ↓
                         DAG 2 (weekly)  →  BigQuery (aggregated / feature tables)
                                              ↓
                         DAG 3 (weekly)  →  XGBoost / LightGBM + MLflow
                                              ↓
                              Streamlit dashboard + Tableau Public
```

All components run in Docker containers defined in `docker-compose.yml`.

## Airflow DAG Structure

| DAG | Schedule | Responsibility |
|-----|----------|----------------|
| DAG 1 | Daily | Generate synthetic users/events/purchases via Faker → upload to GCS |
| DAG 2 | Weekly | Aggregate raw tables into BigQuery feature tables |
| DAG 3 | Weekly | Retrain churn/LTV models, log to MLflow |

## Data Layer

**Raw tables** (GCS, date-partitioned, then loaded to BigQuery): `users`, `unit_stats`, `pilot_stats`, `pilot_unit_synergy`, `gacha_history`, `gacha_event_log`, `stage_clear_log`, `stage_unit_log`, `party_composition_log`, `session_log`, `user_unit_inventory`, `user_daily_snapshot`, `daily_game_metrics`

**Analytics tables** (BigQuery, produced by DAG 2): `events` (user×date aggregation), `purchases` (with cumulative columns), `marketing_events` (CTR/CVR/A-B test data)

`user_daily_snapshot` holds the model-predicted `churn_risk_score` and is recalculated daily.

## ML Models

| Model | Target | Algorithm | Metric |
|-------|--------|-----------|--------|
| Churn Prediction | Binary churn flag | XGBoost / LightGBM | AUC ≥ 0.75 |
| LTV Prediction | 30/90-day revenue | XGBoost Regressor | RMSE, MAE |

## A/B Test Design

- **Control (A):** reward = 1,000 / **Treatment (B):** reward = 1,500 (1.5×)
- Statistical tests: `scipy` t-test + Chi-square (p < 0.05 threshold)
- Causal inference: `DoWhy` / `EconML` for ATE estimation
- Business impact: CVR improvement → projected monthly revenue uplift


## Core Documents
- Full PRD: `Prd 게임유저매출최적화시스템.md`
- Table schema & KPI formulas: `테이블 설계 명세서.md`

Always read both documents before starting any task.

## Key KPIs

`CTR`, `CVR`, `LTV`, Day-1/7/30 Retention Rate, Churn AUC, `DAU/MAU/DNU`, `ARPU`, `ARPPU`, `PUR`, `PCU/ACU`, `ROI`, `ROAS`, `CPM/eCPM`

See `테이블 설계 명세서.md` for full formula definitions and which tables feed each metric.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data generation | `Faker`, `Pandas`, `NumPy` |
| Pipeline | Apache Airflow |
| Storage | GCP Cloud Storage (raw), BigQuery (analytics) |
| ML | `XGBoost`, `LightGBM`, `scikit-learn` |
| Causal inference | `DoWhy`, `EconML` |
| Statistics | `scipy` |
| Experiment tracking | MLflow |
| Dashboard | Streamlit (interactive), Tableau Public (portfolio) |
| Containers | Docker / Docker Compose |
