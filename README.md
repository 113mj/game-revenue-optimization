# Game Revenue Optimization System

모바일 턴제 RPG **"Chronicle Tactics"** 의 유저 데이터를 기반으로 한 엔드투엔드 MLOps 포트폴리오입니다.  
가상 유저 30,000명의 행동 데이터를 자동 생성하고, 데이터 파이프라인 자동화 · 이탈/LTV 예측 모델 · 실시간 대시보드까지 구현합니다.

---

## Architecture

```
[Faker 데이터 생성]
       │
       ▼
  DAG 1 (매일 02:00)          GCS (Raw 데이터 - Parquet 파티션)
  incremental 생성 ──────────►  gs://game-revenue-raw-data-jmj/raw/
       │
       ▼
  DAG 2 (매주 월 03:00)        BigQuery (Analytics 테이블)
  analytics 재계산 ────────────►  game_analytics.events
  BQ 업로드                       game_analytics.purchases
                                  game_analytics.marketing_events
                                  game_analytics.user_daily_snapshot
                                  game_analytics.daily_game_metrics
       │
       ▼
  DAG 3 (매주 월 04:00)        MLflow (실험 추적)
  XGBoost vs LightGBM ────────►  churn_prediction (AUC 비교)
  최고 모델 BQ 반영               ltv_prediction (RMSE 비교)
       │
       ▼
  Streamlit Dashboard          실시간 KPI 시각화
  (localhost:8501)  ◄─────────  BigQuery 직접 조회
```

---

## Key Features

| 구분 | 내용 |
|---|---|
| **데이터 생성** | Faker 기반 16개 테이블, 30,000 유저 × 약 10개월 |
| **파이프라인** | Airflow DAG 3개 (일별/주별 자동 실행) |
| **스토리지** | GCS (raw), BigQuery (analytics) |
| **ML 모델** | 이탈 예측 (XGBoost vs LightGBM, AUC 기준 자동 선택) |
| **LTV 예측** | XGBoost vs LightGBM Regressor (RMSE 기준 자동 선택) |
| **실험 추적** | MLflow — 파라미터·메트릭·모델 버전 관리 |
| **A/B 테스트** | 보상 A(1,000) vs B(1,500) — Chi-square + DoWhy 인과추론(ATE) |
| **설명 가능한 AI** | SHAP Feature Importance — 이탈 예측 주요 요인 시각화 |
| **코호트 분석** | Day-1 / Day-7 / Day-30 리텐션 히트맵 |
| **AI 에이전트** | Gemini 2.5 Flash 기반 자연어 → BigQuery SQL 자동 생성 |
| **대시보드** | Streamlit 5탭 — Overview / A/B Test / Churn·LTV / Funnel / AI 에이전트 |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Generation | Python, Faker, Pandas, NumPy |
| Orchestration | Apache Airflow 2.9 (LocalExecutor) |
| Storage | GCP Cloud Storage, BigQuery |
| ML | XGBoost, LightGBM, scikit-learn |
| Causal Inference | DoWhy, statsmodels |
| Explainability | SHAP |
| Experiment Tracking | MLflow 2.14 |
| Dashboard | Streamlit, Plotly |
| AI Agent | Google Gemini 2.5 Flash (Function Calling) |
| Infrastructure | Docker, Docker Compose |

---

## Project Structure

```
game-revenue-optimization/
├── dags/
│   ├── dag1_daily_pipeline.py   # 매일: 데이터 생성 → GCS
│   ├── dag2_bq_aggregation.py   # 매주: analytics 재계산 → BigQuery
│   └── dag3_model_training.py   # 매주: XGBoost/LightGBM 학습 → MLflow
├── data_generator/
│   ├── config.py                # GCP 설정, 상수 정의
│   ├── generate_users.py        # 유저 테이블 (30k명)
│   ├── generate_meta.py         # 유닛/파일럿/가챠 메타
│   ├── generate_daily_raw.py    # 세션/가챠/스테이지 로그
│   ├── generate_analytics.py    # events/purchases/marketing_events
│   ├── generate_snapshots.py    # user_daily_snapshot/daily_game_metrics
│   ├── generate_inventory.py    # 유저 인벤토리
│   ├── main.py                  # 실행 진입점 (backfill / incremental)
│   └── upload_to_gcp.py         # GCS/BQ 업로드
├── dashboard/
│   └── app.py                   # Streamlit 대시보드 (5탭 + Gemini AI 에이전트)
└── docker-compose.yml           # 전체 서비스 정의
```

---

## Data Design

**Raw 테이블** (GCS → BigQuery)

| 테이블 | 설명 |
|---|---|
| users | 유저 기본 정보, A/B 그룹 배정 |
| unit_stats / pilot_stats | 게임 캐릭터 메타 |
| gacha_history / gacha_event_log | 뽑기 로그 |
| session_log | 세션 기록 |
| stage_clear_log / stage_unit_log | 스테이지 클리어 |
| user_unit_inventory | 유저 보유 캐릭터 |
| user_daily_snapshot | 유저별 일별 누적 지표 + **churn_risk_score** |
| daily_game_metrics | DAU/MAU/DNU/ARPU/ARPPU/PUR |

**Analytics 테이블** (BigQuery, DAG2 생성)

| 테이블 | 설명 |
|---|---|
| events | 유저×날짜 행동 집계 |
| purchases | 결제 로그 (누적 LTV 포함) |
| marketing_events | A/B 그룹별 노출·클릭·전환 |

---

## ML Pipeline

### 이탈 예측 (Churn Prediction)
- **Target**: 최근 14일 미접속 → 이탈 (binary)
- **Features**: 레벨, 플레이타임, 가챠 횟수, 결제금액, 활성일수, 평균 세션 등
- **모델 비교**: XGBoost vs LightGBM → AUC 높은 모델 자동 선택
- **목표 성능**: AUC ≥ 0.75

### LTV 예측 (LTV Prediction)
- **Target**: 누적 결제 금액 (continuous)
- **Features**: 레벨, 플레이타임, 가챠 횟수, 결제 횟수, 평균 결제금액
- **모델 비교**: XGBoost vs LightGBM → RMSE 낮은 모델 자동 선택

### 모델 자동 배포 흐름
```
DAG3 실행 → 두 모델 학습 → MLflow 메트릭 비교
         → 최고 모델 선택 → BigQuery churn_risk_score 갱신
         → Streamlit 대시보드 자동 반영
```

---

## A/B Test Design

| 그룹 | 보상 금액 | CVR 설계 |
|---|---|---|
| A (Control) | 1,000 | 3.0% |
| B (Treatment) | 1,500 | 4.5% |

- 통계 검정: Chi-square test (p < 0.05 기준)
- 인과추론: DoWhy ATE(Average Treatment Effect) 추정 + Placebo 반박 검정
- 비즈니스 임팩트: 월간 매출 증가 추정

---

## KPIs Tracked

`DAU` · `MAU` · `DNU` · `ARPU` · `ARPPU` · `PUR` · `PCU` · `ACU`  
`CTR` · `CVR` · `Churn AUC` · `LTV RMSE`

---

## Quick Start

### 1. 사전 준비
```bash
# GCP 서비스 계정 키 준비 후 .env 설정
cp .env.example .env
# .env에 GCP_PROJECT_ID, GCS_BUCKET_NAME, BIGQUERY_DATASET, GOOGLE_APPLICATION_CREDENTIALS 입력
```

### 2. 초기 데이터 생성 및 GCP 업로드
```bash
cd data_generator
pip install -r ../requirements.txt
python main.py backfill          # 전체 기간 데이터 생성
python upload_to_gcp.py --gcs   # GCS 업로드
python upload_to_gcp.py --bq    # BigQuery 업로드
```

### 3. 서비스 실행
```bash
docker-compose up -d
```

| 서비스 | URL |
|---|---|
| Airflow UI | http://localhost:8080 (admin/admin) |
| MLflow UI | http://localhost:5001 |
| Streamlit Dashboard | http://localhost:8501 |

### 4. DAG 실행
Airflow UI에서 DAG1 → DAG2 → DAG3 순으로 수동 트리거하거나, 스케줄에 따라 자동 실행됩니다.

---

## Environment Variables

| 변수 | 설명 |
|---|---|
| `GCP_PROJECT_ID` | GCP 프로젝트 ID |
| `GCS_BUCKET_NAME` | GCS 버킷 이름 |
| `BIGQUERY_DATASET` | BigQuery 데이터셋 이름 |
| `GOOGLE_APPLICATION_CREDENTIALS` | 서비스 계정 JSON 경로 |
| `MLFLOW_TRACKING_URI` | MLflow 서버 URI (기본: http://mlflow:5000) |
| `GEMINI_API_KEY` | Google Gemini API 키 ([발급](https://aistudio.google.com/app/apikey)) |
