"""
공통 설정 및 상수 정의.
.env 파일에서 환경변수 로드.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / ".env")

# ── GCP ──────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID",  "game-revenue-optimization")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "game-revenue-raw-data-jmj")
BQ_DATASET      = os.getenv("BIGQUERY_DATASET", "game_analytics")

# ── 날짜 ─────────────────────────────────────────────────────────────────────
START_DATE = date(2025, 7, 1)
TODAY      = date.today()
YESTERDAY  = TODAY - timedelta(days=1)

# ── 규모 ─────────────────────────────────────────────────────────────────────
N_USERS        = 30_000
N_UNITS        = 200
N_PILOTS       = 200
N_STAGES       = 200
N_GACHA_EVENTS = 50   # 픽업 이벤트 수 (전체 기간)

# ── 유저 속성 ─────────────────────────────────────────────────────────────────
COUNTRIES       = ["KR", "US", "JP", "TW", "TH"]
COUNTRY_WEIGHTS = [0.40, 0.20, 0.20, 0.10, 0.10]

DEVICES        = ["ios", "android"]
DEVICE_WEIGHTS = [0.45, 0.55]

CHANNELS        = ["organic", "paid", "sns", "store"]
CHANNEL_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

# ── 게임 콘텐츠 ───────────────────────────────────────────────────────────────
GRADES              = ["N", "R", "SR", "SSR"]
GACHA_NORMAL_WEIGHTS   = [0.60, 0.25, 0.12, 0.03]
GACHA_PREMIUM_WEIGHTS  = [0.40, 0.30, 0.20, 0.10]
GACHA_EVENT_WEIGHTS    = [0.30, 0.30, 0.25, 0.15]

UNIT_TYPES   = ["공격", "방어", "지원", "균형"]
UNIT_SERIES  = ["Series_A", "Series_B", "Series_C", "Series_D", "Series_E"]
PILOT_SKILLS = ["공격강화", "방어강화", "회복", "속도증가", "크리티컬강화"]
SYNERGY_GRADES = ["S", "A", "B", "C"]
SYNERGY_GRADE_WEIGHTS = [0.10, 0.25, 0.40, 0.25]
STRONG_STAGE_TYPES = ["스토리", "레이드", "PvP", "이벤트"]
ITEM_CATEGORIES = ["gacha", "costume", "stamina", "package"]
ITEM_CATEGORY_WEIGHTS = [0.45, 0.20, 0.25, 0.10]

# ── 비즈니스 파라미터 ──────────────────────────────────────────────────────────
CHURN_RATE       = 0.65   # 전체 유저 중 이탈 비율
PAYING_USER_RATE = 0.10   # 결제 경험 유저 비율
CVR_A            = 0.030  # A그룹 전환율
CVR_B            = 0.045  # B그룹 전환율
CTR_BASE         = 0.15   # 기본 클릭률

DAU_RATE         = 0.30   # 활성 유저 중 일별 로그인 비율 (기대치)
AVG_SESSION_MIN  = 35     # 평균 세션 시간(분)

# 결제 금액 분포 (원)
PURCHASE_AMOUNTS = [990, 1900, 4900, 9900, 29900, 49900, 99900]
PURCHASE_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02]

# ── 로컬 출력 경로 ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parents[1] / "data"

RANDOM_SEED = 42
