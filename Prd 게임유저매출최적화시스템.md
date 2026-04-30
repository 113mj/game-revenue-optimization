# 📄 PRD: 게임 유저 매출 최적화 시스템

**버전:** v2.0  
**작성일:** 2026-04-14  
**타겟 포지션:** 마케팅 전략 데이터 사이언티스트 / AD-Tech 데이터 사이언티스트

---

## 1. 프로젝트 개요

### 1.1 프로젝트명
Game User Revenue Optimization System

### 1.2 목표
가상의 게임 유저 데이터를 기반으로 CTR, CVR, LTV를 분석하고, A/B 테스트 및 인과추론을 통해 유저 유지율과 매출을 최적화하는 데이터 기반 의사결정 시스템 구축

### 1.3 핵심 비즈니스 스토리 (면접용)
> "보상 A/B 테스트 결과 CVR이 통계적으로 유의미하게 상승했고, 이탈 위험 유저를 사전에 식별해 타겟 마케팅 전략을 제안했다. 이탈 위험 유저 상위 20%에 리텐션 보상을 줬을 때 예상 매출 회복액을 정량화했다."

### 1.4 기대 효과
- 유저 이탈 감소 및 결제 전환율 증가
- 고가치 유저(High LTV) 식별 및 세그먼트 전략
- 마케팅 전략의 인과적 효과 검증
- 데이터 기반 의사결정 파이프라인 자동화

---

## 2. 타겟 공고 분석

### 2.1 마케팅 전략 데이터 사이언티스트 (메인 타겟)
| 요구 기술 | 프로젝트 적용 |
|-----------|--------------|
| Python (Pandas, NumPy, scikit-learn, XGBoost, LightGBM) | 데이터 생성 + 모델링 |
| SQL, Apache Spark | BigQuery SQL 분석 |
| Git, Docker, Kubernetes | Docker 컨테이너화 |
| Kubeflow Pipelines, Apache Airflow | Airflow DAG 3종 |
| GCP (Vertex AI, BigQuery, Cloud Storage, Dataflow) | GCP 풀스택 |
| Tableau, Looker, Power BI | Tableau Public |

### 2.2 AD-Tech 데이터 사이언티스트 (추가 타겟)
| 요구 기술 | 프로젝트 적용 |
|-----------|--------------|
| Python | 전체 파이프라인 |
| ML/통계 프레임워크 (LightGBM, XGBoost) | 이탈 예측 + LTV 예측 |
| SQL, Apache Spark | BigQuery SQL |
| Experimentation & Causal Inference (DoWhy, EconML) | A/B 테스트 + 인과추론 |

---

## 3. 사용자 페르소나

### 3.1 게임 데이터 분석가
- 유저 행동 분석 및 퍼널 이탈 구간 파악
- A/B 테스트 결과 기반 의사결정

### 3.2 마케팅 담당자
- 이벤트 효과 측정 및 인과적 효과 검증
- 이탈 위험 유저 타겟 선정 및 보상 전략 수립

---

## 4. 기능 명세

### 4.1 데이터 파이프라인 자동화 (Airflow)
- 유저 / 이벤트 / 구매 데이터 일별 자동 생성 및 GCS 적재
- 주간 집계 및 피처 테이블 BigQuery 업로드
- 주간 모델 재학습 트리거 및 MLflow 로깅

### 4.2 퍼널 분석 (CTR / CVR)
- 노출 → 클릭 → 결제 퍼널 구성
- CTR, CVR 계산 및 단계별 이탈률 시각화

### 4.3 코호트 분석 (Retention)
- Day 1 / Day 7 / Day 30 유지율 분석
- 가입 채널별 코호트 비교

### 4.4 유저 세그먼트 분석
- 고가치 유저 (High LTV)
- 일반 유저
- 이탈 위험 유저 (Churn Risk)

### 4.5 ML 모델링

#### 이탈 예측 (Churn Prediction)
- 입력: 유저 행동 데이터 (플레이타임, 로그인 빈도, 레벨 등)
- 출력: 이탈 확률 스코어
- 모델: XGBoost / LightGBM
- 평가: AUC-ROC, F1

#### LTV 예측
- 입력: 유저 활동 및 결제 데이터
- 출력: 예상 수익 (30일 / 90일)
- 모델: XGBoost Regressor
- 평가: RMSE, MAE

### 4.6 A/B 테스트 + 인과추론 (핵심 차별화)
- 가설: 보상 증가가 전환율을 높일 것이다
- A 그룹: 기존 보상 / B 그룹: 보상 증가
- 통계 검정: t-test, Chi-square (scipy)
- 인과추론: DoWhy / EconML로 Average Treatment Effect(ATE) 추정
- 비즈니스 임팩트: CVR 개선 → 예상 매출 증가액 정량화

### 4.7 전략 추천 시스템
- 이탈 위험 유저 상위 20% → 리텐션 보상 지급 추천 + 예상 매출 회복액
- 고 LTV 유저 → VIP 이벤트 추천
- 비활성 유저 → 복귀 프로모션 추천

### 4.8 시각화 대시보드

#### Streamlit (인터랙션용)
- 실시간 퍼널 분석 차트
- 이탈 예측 모델 인터랙션 (유저 ID 입력 → 이탈 확률 출력)
- A/B 테스트 결과 요약

#### Tableau Public (포트폴리오용)
- 코호트 차트
- LTV 분포
- 유저 세그먼트 맵

---

## 5. 데이터 설계

### 5.1 users 테이블
| 컬럼 | 타입 | 설명 |
|------|------|------|
| user_id | STRING | 유저 고유 ID |
| signup_date | DATE | 가입일 |
| country | STRING | 국가 |
| acquisition_channel | STRING | 유입 채널 (organic/paid/sns) |
| ab_group | STRING | A/B 그룹 (A/B) |

### 5.2 events 테이블
| 컬럼 | 타입 | 설명 |
|------|------|------|
| user_id | STRING | 유저 ID |
| event_date | DATE | 이벤트 날짜 |
| playtime_minutes | INT | 플레이 시간 |
| level | INT | 현재 레벨 |
| login_yn | BOOL | 로그인 여부 |

### 5.3 purchases 테이블
| 컬럼 | 타입 | 설명 |
|------|------|------|
| user_id | STRING | 유저 ID |
| purchase_date | DATE | 결제일 |
| purchase_amount | FLOAT | 결제 금액 |
| item_category | STRING | 아이템 카테고리 |

### 5.4 marketing_events 테이블
| 컬럼 | 타입 | 설명 |
|------|------|------|
| user_id | STRING | 유저 ID |
| impression_yn | BOOL | 노출 여부 |
| click_yn | BOOL | 클릭 여부 |
| conversion_yn | BOOL | 전환 여부 |
| campaign_date | DATE | 캠페인 날짜 |

---

## 6. 시스템 아키텍처

```
[Python + Faker]
      ↓ (Airflow DAG 1 - 매일)
[GCS - 원본 데이터 저장]
      ↓ (Airflow DAG 2 - 주간)
[BigQuery - 집계 / 피처 테이블]
      ↓ (Airflow DAG 3 - 주간)
[XGBoost / LightGBM + MLflow]
      ↓
[Streamlit 대시보드 + Tableau Public]
```

모든 컴포넌트는 Docker로 컨테이너화

---

## 7. 기술 스택

| 레이어 | 기술 |
|--------|------|
| 언어 | Python, SQL |
| 데이터 생성 | Faker, Pandas, NumPy |
| 파이프라인 | Apache Airflow |
| 저장소 | GCP Cloud Storage, BigQuery |
| ML 모델링 | XGBoost, LightGBM, Scikit-learn |
| 인과추론 | DoWhy / EconML |
| 통계 검정 | Scipy |
| 실험 관리 | MLflow |
| 컨테이너 | Docker |
| 대시보드 | Streamlit, Tableau Public |
| 버전 관리 | Git / GitHub |

---

## 8. 핵심 지표 (KPI)

| 지표 | 설명 | 목표 |
|------|------|------|
| CTR | 노출 대비 클릭률 | A/B 비교 |
| CVR | 클릭 대비 전환율 | B그룹 통계적 유의미한 개선 |
| LTV | 유저 생애 가치 예측 | RMSE 기준 성능 확보 |
| Retention Rate | Day 1/7/30 유지율 | 코호트별 비교 |
| Churn AUC | 이탈 예측 모델 성능 | AUC ≥ 0.75 |

---

## 9. A/B 테스트 설계

### 9.1 가설
귀무가설 H₀: 보상 증가는 CVR에 영향을 미치지 않는다  
대립가설 H₁: 보상 증가는 CVR을 유의미하게 높인다

### 9.2 실험 방법
- A 그룹: 기존 보상 (50% 유저)
- B 그룹: 보상 1.5배 증가 (50% 유저)
- 실험 기간: 2주

### 9.3 분석 방법
1. Scipy t-test / Chi-square로 통계적 유의성 검정 (p < 0.05)
2. DoWhy/EconML로 ATE(Average Treatment Effect) 추정
3. CVR 개선 → 예상 월 매출 증가액 계산 (비즈니스 임팩트 정량화)

---

## 10. 1달 개발 타임라인

| 주차 | 작업 내용 |
|------|-----------|
| 1주차 | 데이터 스키마 확정 + Faker 기반 가상 데이터 생성 + EDA |
| 2주차 | Airflow DAG 3종 구현 + GCS/BigQuery 연동 + 퍼널/코호트 분석 |
| 3주차 | A/B 테스트 + DoWhy 인과추론 + 이탈/LTV 모델 + MLflow |
| 4주차 | Streamlit 대시보드 + Tableau 시각화 + README + GitHub 정리 |

---

## 11. 성공 기준

- Airflow DAG 3종 정상 작동 (스케줄 자동 실행)
- A/B 테스트 p-value < 0.05 달성 (또는 분석적 인사이트 도출)
- 이탈 예측 AUC ≥ 0.75
- LTV 예측 성능 확보 (RMSE 기준)
- Streamlit + Tableau 대시보드 완성
- GitHub README에 비즈니스 임팩트 숫자 명시

---

## 12. 향후 확장 (Future Work)

- 실시간 데이터 처리 (Kafka / Dataflow)
- 개인화 추천 시스템 고도화
- 자동 A/B 테스트 시스템
- Vertex AI 기반 모델 서빙
- 강화학습 기반 마케팅 최적화

---

## 최종 요약

본 프로젝트는 게임 유저 데이터 파이프라인(Airflow + GCP)을 자동화하고, 퍼널 분석 / 코호트 분석 / A/B 테스트 + 인과추론 / 이탈 예측 / LTV 예측을 수행하여 데이터 기반 마케팅 전략을 도출하는 MLOps 통합 시스템이다.

**타겟 공고:** 마케팅 전략 DS + AD-Tech DS  
**핵심 차별화:** DoWhy/EconML 인과추론 + 비즈니스 임팩트 정량화  
**MLOps 경력 활용:** Airflow DAG 자동화 + MLflow 실험 관리 + Docker 컨테이너화