"""
Chronicle Tactics — Game Revenue Optimization Dashboard
BigQuery 기반 KPI / A/B Test / Churn / Funnel 분석
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.cloud import bigquery
from scipy import stats

# ── 설정 ──────────────────────────────────────────────────────────────────────
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "game-revenue-optimization")
DATASET    = os.getenv("BIGQUERY_DATASET", "game_analytics")
P          = f"{PROJECT_ID}.{DATASET}"

st.set_page_config(
    page_title="Chronicle Tactics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎮 Chronicle Tactics — Revenue Optimization Dashboard")
st.caption(f"Data: BigQuery `{P}`")

# ── BQ 클라이언트 ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return bigquery.Client(project=PROJECT_ID)

@st.cache_data(ttl=3600, show_spinner="데이터 불러오는 중...")
def bq(sql: str) -> pd.DataFrame:
    return get_client().query(sql).to_dataframe()

# ── 사이드바 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 필터")
    n_days = st.slider("조회 기간 (최근 N일)", 7, 270, 90)
    st.caption(f"현재 설정: 최근 {n_days}일")
    st.markdown("---")
    st.markdown("**데이터 새로고침**")
    if st.button("🔄 캐시 초기화"):
        st.cache_data.clear()
        st.rerun()

# ── 탭 ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🧪 A/B Test",
    "🔮 Churn / LTV",
    "🔽 Funnel & Retention",
    "🤖 AI 에이전트",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    df = bq(f"""
        SELECT *
        FROM `{P}.daily_game_metrics`
        WHERE metric_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n_days} DAY)
        ORDER BY metric_date
    """)

    if df.empty:
        st.warning("데이터가 없습니다.")
    else:
        df["metric_date"] = pd.to_datetime(df["metric_date"])
        recent7 = df.tail(7)

        # KPI 카드
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("DAU (7일 평균)",   f"{int(recent7['DAU'].mean()):,}")
        c2.metric("MAU (최근)",        f"{int(df['MAU'].iloc[-1]):,}")
        c3.metric("ARPU (7일 평균)",   f"₩{recent7['ARPU'].mean():,.1f}")
        c4.metric("ARPPU (7일 평균)",  f"₩{recent7['ARPPU'].mean():,.0f}")
        c5.metric("PUR (7일 평균)",    f"{recent7['PUR'].mean()*100:.2f}%")

        st.markdown("---")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig = px.area(
                df, x="metric_date", y="total_revenue",
                title="일별 매출",
                labels={"total_revenue": "매출 (₩)", "metric_date": ""},
            )
            fig.update_traces(fillcolor="rgba(255,107,107,0.2)", line_color="#FF6B6B")
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            fig = px.line(
                df, x="metric_date", y=["DAU", "DNU"],
                title="DAU / DNU 추이",
                labels={"value": "유저 수", "metric_date": "", "variable": "지표"},
            )
            st.plotly_chart(fig, use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig = px.area(
                df, x="metric_date", y="MAU",
                title="MAU 추이",
                labels={"MAU": "월간 활성 유저", "metric_date": ""},
            )
            fig.update_traces(fillcolor="rgba(78,205,196,0.2)", line_color="#4ECDC4")
            st.plotly_chart(fig, use_container_width=True)

        with r2c2:
            fig = px.line(
                df, x="metric_date", y=["ARPU", "ARPPU"],
                title="ARPU / ARPPU",
                labels={"value": "₩", "metric_date": "", "variable": "지표"},
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  A/B Test
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    ab_df = bq(f"""
        SELECT
            CASE WHEN reward_amount = 1500 THEN 'B (보상 1,500)' ELSE 'A (보상 1,000)' END AS ab_group,
            COUNT(*)                           AS impressions,
            COUNTIF(click_yn)                  AS clicks,
            COUNTIF(conversion_yn)             AS conversions,
            ROUND(SUM(ad_revenue), 2)          AS total_revenue
        FROM `{P}.marketing_events`
        WHERE campaign_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n_days} DAY)
        GROUP BY ab_group
        ORDER BY ab_group
    """)

    if ab_df.empty or len(ab_df) < 2:
        st.warning("A/B 테스트 데이터가 부족합니다.")
    else:
        ab_df["CTR"] = ab_df["clicks"]       / ab_df["impressions"]
        ab_df["CVR"] = ab_df["conversions"]  / ab_df["clicks"].replace(0, np.nan)

        a = ab_df.iloc[0]
        b = ab_df.iloc[1]

        st.subheader("그룹별 핵심 지표")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CTR (A)",  f"{a.CTR*100:.2f}%")
        c2.metric("CTR (B)",  f"{b.CTR*100:.2f}%",  f"{(b.CTR-a.CTR)/a.CTR*100:+.1f}%")
        c3.metric("CVR (A)",  f"{a.CVR*100:.2f}%")
        c4.metric("CVR (B)",  f"{b.CVR*100:.2f}%",  f"{(b.CVR-a.CVR)/a.CVR*100:+.1f}%")

        st.markdown("---")
        st.subheader("통계적 유의성 (Chi-square test)")

        ctr_chi2, ctr_p, _, _ = stats.chi2_contingency([
            [int(a.clicks),      int(a.impressions - a.clicks)],
            [int(b.clicks),      int(b.impressions - b.clicks)],
        ])
        cvr_chi2, cvr_p, _, _ = stats.chi2_contingency([
            [int(a.conversions), int(a.clicks - a.conversions)],
            [int(b.conversions), int(b.clicks - b.conversions)],
        ])

        col1, col2 = st.columns(2)
        for col, label, chi2, p in [
            (col1, "CTR", ctr_chi2, ctr_p),
            (col2, "CVR", cvr_chi2, cvr_p),
        ]:
            with col:
                st.markdown(f"**{label} 검정**")
                st.markdown(f"χ² = `{chi2:.4f}` &nbsp;&nbsp; p-value = `{p:.4f}`")
                if p < 0.05:
                    st.success("✅ 통계적으로 유의미 (p < 0.05)")
                else:
                    st.warning("⚠️ 유의미하지 않음 (p ≥ 0.05)")

        st.markdown("---")

        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.bar(
                ab_df.melt(id_vars="ab_group", value_vars=["CTR", "CVR"]),
                x="variable", y="value", color="ab_group", barmode="group",
                title="CTR / CVR 비교",
                labels={"variable": "지표", "value": "비율", "ab_group": "그룹"},
                color_discrete_sequence=["#4ECDC4", "#FF6B6B"],
            )
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("💰 월간 매출 임팩트 추정")
            daily_imp = int((a.impressions + b.impressions) / n_days)
            rpc_a = a.total_revenue / max(a.conversions, 1)
            rpc_b = b.total_revenue / max(b.conversions, 1)
            rev_a = daily_imp * a.CTR * a.CVR * rpc_a
            rev_b = daily_imp * b.CTR * b.CVR * rpc_b
            uplift = (rev_b - rev_a) * 30

            st.metric("A그룹 일 예상 매출",  f"₩{rev_a:,.0f}")
            st.metric("B그룹 일 예상 매출",  f"₩{rev_b:,.0f}")
            st.metric("월간 예상 매출 증가",  f"₩{uplift:,.0f}",
                      f"{(rev_b - rev_a) / max(rev_a, 1) * 100:+.1f}%")

    # ── DoWhy 인과 추론 ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 인과 추론 (DoWhy — ATE 추정)")
    st.caption("Chi-square는 '차이가 있는가'를 검정합니다. DoWhy는 '보상 정책이 전환율을 얼마나 인과적으로 변화시켰는가'를 추정합니다.")

    if st.button("▶ 인과 분석 실행", key="run_dowhy"):
        with st.spinner("DoWhy 분석 중... (30초~1분 소요)"):
            try:
                from dowhy import CausalModel

                causal_df = bq(f"""
                    SELECT
                        CASE WHEN reward_amount = 1500 THEN 1 ELSE 0 END AS treatment,
                        CAST(conversion_yn AS INT64) AS outcome
                    FROM `{P}.marketing_events`
                    WHERE campaign_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n_days} DAY)
                """)

                if causal_df.empty:
                    st.warning("데이터가 없습니다.")
                else:
                    if len(causal_df) > 50000:
                        causal_df = causal_df.sample(50000, random_state=42)

                    model = CausalModel(
                        data=causal_df,
                        treatment="treatment",
                        outcome="outcome",
                        graph="digraph {treatment -> outcome;}",
                    )

                    estimand = model.identify_effect(proceed_when_unidentifiable=True)
                    estimate = model.estimate_effect(
                        estimand,
                        method_name="backdoor.linear_regression",
                        test_significance=True,
                    )
                    ate = float(estimate.value)

                    # 반증 검정: Placebo Treatment
                    refute = model.refute_estimate(
                        estimand,
                        estimate,
                        method_name="placebo_treatment_refuter",
                        placebo_type="permute",
                        num_simulations=20,
                    )
                    placebo_ate = float(refute.new_effect)

                    # 결과 표시
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("ATE (평균 처리 효과)", f"{ate*100:.3f}%p")
                    rc2.metric("Placebo ATE", f"{placebo_ate*100:.3f}%p")
                    rc3.metric("해석",
                               "인과 효과 유효" if abs(placebo_ate) < abs(ate) * 0.2
                               else "효과 불안정")

                    st.markdown(f"""
                    **결과 해석**
                    - 보상 1,500 정책(B그룹)은 A그룹 대비 전환율을 **{ate*100:.3f}%p** 인과적으로 상승시킴
                    - Placebo ATE = **{placebo_ate*100:.3f}%p** (무작위 처치 시 효과 → 0에 가까울수록 원래 추정이 견고함)
                    """)

                    if abs(placebo_ate) < abs(ate) * 0.2:
                        st.success("✅ Placebo ATE ≈ 0 — 인과 효과가 통계적으로 견고합니다")
                    else:
                        st.warning("⚠️ Placebo ATE가 크게 나왔습니다. 추정 결과를 신중히 해석하세요")

            except ImportError:
                st.error("DoWhy 패키지가 설치되지 않았습니다. requirements.txt를 확인해주세요.")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  Churn / LTV
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    snap = bq(f"""
        SELECT
            user_id,
            level,
            total_playtime,
            total_gacha,
            total_spent,
            churn_risk_score,
            CASE
                WHEN churn_risk_score >= 0.7 THEN 'High (≥0.7)'
                WHEN churn_risk_score >= 0.4 THEN 'Medium (0.4–0.7)'
                ELSE 'Low (<0.4)'
            END AS risk_level
        FROM `{P}.user_daily_snapshot`
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date) FROM `{P}.user_daily_snapshot`
        )
    """)

    if snap.empty:
        st.warning("스냅샷 데이터 없음")
    else:
        high = snap[snap.risk_level == "High (≥0.7)"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 유저",        f"{len(snap):,}")
        c2.metric("이탈 고위험",      f"{len(high):,}",
                  f"{len(high)/len(snap)*100:.1f}%")
        c3.metric("평균 이탈 스코어",  f"{snap.churn_risk_score.mean():.3f}")
        c4.metric("유저 평균 결제액",  f"₩{snap.total_spent.mean():,.0f}")

        st.markdown("---")

        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.histogram(
                snap, x="churn_risk_score", nbins=40, color="risk_level",
                color_discrete_map={
                    "High (≥0.7)":      "#FF6B6B",
                    "Medium (0.4–0.7)": "#FFD93D",
                    "Low (<0.4)":       "#6BCB77",
                },
                title="이탈 위험 스코어 분포",
                labels={"churn_risk_score": "스코어", "count": "유저 수"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            sample = snap.sample(min(3000, len(snap)), random_state=42)
            fig = px.scatter(
                sample, x="total_spent", y="churn_risk_score",
                color="risk_level",
                color_discrete_map={
                    "High (≥0.7)":      "#FF6B6B",
                    "Medium (0.4–0.7)": "#FFD93D",
                    "Low (<0.4)":       "#6BCB77",
                },
                title="누적 결제액 vs 이탈 위험",
                labels={"total_spent": "누적 결제액 (₩)", "churn_risk_score": "이탈 스코어"},
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("세그먼트별 요약")
        seg = (
            snap.groupby("risk_level")
            .agg(
                유저수=("user_id", "count"),
                평균결제액=("total_spent", "mean"),
                평균레벨=("level", "mean"),
                평균플레이타임=("total_playtime", "mean"),
            )
            .round(1)
            .reset_index()
            .rename(columns={"risk_level": "위험 등급"})
        )
        st.dataframe(seg, use_container_width=True, hide_index=True)

        st.subheader("⚠️ 이탈 고위험 유저 Top 20")
        top = (
            snap.nlargest(20, "churn_risk_score")[
                ["user_id", "churn_risk_score", "total_spent", "level", "total_playtime"]
            ]
            .rename(columns={
                "user_id":          "유저 ID",
                "churn_risk_score": "이탈 스코어",
                "total_spent":      "누적 결제액",
                "level":            "레벨",
                "total_playtime":   "총 플레이타임",
            })
            .reset_index(drop=True)
        )
        st.dataframe(top, use_container_width=True, hide_index=True)

        # ── SHAP 피처 중요도 ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 SHAP 피처 중요도")
        st.caption("이탈 예측 모델이 어떤 요인을 가장 중요하게 보는지 설명합니다.")

        CHURN_FEATURES = [
            "max_level", "total_playtime", "total_gacha", "total_spent",
            "active_days", "avg_daily_playtime", "avg_sessions", "max_consec_login",
        ]
        FEATURE_KO = {
            "max_level":          "최고 레벨",
            "total_playtime":     "총 플레이타임",
            "total_gacha":        "총 가챠 횟수",
            "total_spent":        "누적 결제금액",
            "active_days":        "활성 일수",
            "avg_daily_playtime": "일 평균 플레이타임",
            "avg_sessions":       "일 평균 세션 수",
            "max_consec_login":   "최대 연속 로그인",
        }

        try:
            import shap
            import json
            from pathlib import Path as FPath
            from xgboost import XGBClassifier

            model_dir = FPath("/dashboard/data/models")
            feat_path = FPath("/dashboard/data/features/churn_features.parquet")

            if not feat_path.exists():
                st.warning("모델 파일이 없습니다. DAG3를 먼저 실행해주세요.")
            else:
                feat_df = pd.read_parquet(feat_path).dropna(subset=CHURN_FEATURES)
                sample  = feat_df[CHURN_FEATURES].sample(min(2000, len(feat_df)), random_state=42)

                best_model = "xgb"
                meta_path  = model_dir / "churn_meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        best_model = json.load(f).get("best_churn_model", "xgb")

                try:
                    if best_model == "lgbm" and (model_dir / "churn_lgbm.txt").exists():
                        import lightgbm as lgb
                        booster   = lgb.Booster(model_file=str(model_dir / "churn_lgbm.txt"))
                        explainer = shap.TreeExplainer(booster)
                    else:
                        raise ImportError
                except Exception:
                    model = XGBClassifier()
                    model.load_model(str(model_dir / "churn_xgb.json"))
                    explainer  = shap.TreeExplainer(model)
                    best_model = "xgb"

                shap_vals = explainer.shap_values(sample)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

                importance_df = pd.DataFrame({
                    "feature":    [FEATURE_KO.get(f, f) for f in CHURN_FEATURES],
                    "importance": np.abs(shap_vals).mean(axis=0),
                }).sort_values("importance", ascending=True)

                fig = px.bar(
                    importance_df, x="importance", y="feature",
                    orientation="h",
                    title=f"SHAP 피처 중요도 (모델: {best_model.upper()})",
                    labels={"importance": "Mean |SHAP value|", "feature": "피처"},
                    color="importance",
                    color_continuous_scale="Reds",
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

                top_feat = importance_df.iloc[-1]["feature"]
                st.info(f"**가장 중요한 이탈 요인**: {top_feat} — 이 지표가 낮을수록 이탈 가능성이 높습니다.")

        except ImportError:
            st.warning("SHAP 패키지가 설치되지 않았습니다.")
        except Exception as e:
            st.error(f"SHAP 분석 오류: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  Funnel & Retention
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    funnel = bq(f"""
        WITH
        total  AS (SELECT COUNT(DISTINCT user_id) AS n FROM `{P}.user_daily_snapshot`),
        active AS (SELECT COUNT(DISTINCT user_id) AS n FROM `{P}.events`
                   WHERE event_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n_days} DAY)),
        paying AS (SELECT COUNT(DISTINCT user_id) AS n FROM `{P}.purchases`
                   WHERE purchase_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n_days} DAY))
        SELECT '① 전체 유저'  AS stage, (SELECT n FROM total)  AS users
        UNION ALL
        SELECT '② 활성 유저'  AS stage, (SELECT n FROM active) AS users
        UNION ALL
        SELECT '③ 결제 유저'  AS stage, (SELECT n FROM paying) AS users
    """)

    cat_df = bq(f"""
        SELECT
            item_category,
            COUNT(*)                          AS purchase_count,
            ROUND(SUM(purchase_amount), 0)    AS total_revenue
        FROM `{P}.purchases`
        WHERE purchase_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n_days} DAY)
        GROUP BY item_category
        ORDER BY total_revenue DESC
    """)
    if not cat_df.empty:
        cat_df = cat_df.rename(columns={"purchase_count": "구매건수", "total_revenue": "총매출"})

    col_l, col_r = st.columns(2)
    with col_l:
        if not funnel.empty:
            fig = go.Figure(go.Funnel(
                y=funnel["stage"].tolist(),
                x=funnel["users"].tolist(),
                textinfo="value+percent initial",
                marker=dict(color=["#4ECDC4", "#45B7D1", "#96CEB4"]),
            ))
            fig.update_layout(title="유저 전환 퍼널")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if not cat_df.empty:
            fig = px.pie(
                cat_df, names="item_category", values="총매출",
                title="카테고리별 매출 비중",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader(f"일별 결제 전환율 (PUR) — 최근 {n_days}일")

    pur_df = bq(f"""
        SELECT
            e.event_date,
            COUNT(DISTINCT e.user_id)                         AS dau,
            COUNT(DISTINCT p.user_id)                         AS paying_users,
            SAFE_DIVIDE(COUNT(DISTINCT p.user_id),
                        COUNT(DISTINCT e.user_id))            AS pur
        FROM `{P}.events` e
        LEFT JOIN `{P}.purchases` p
            ON e.user_id = p.user_id
           AND e.event_date = p.purchase_date
        WHERE e.event_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {n_days} DAY)
        GROUP BY e.event_date
        ORDER BY e.event_date
    """)

    if not pur_df.empty:
        pur_df["event_date"] = pd.to_datetime(pur_df["event_date"])
        fig = px.line(
            pur_df, x="event_date", y="pur",
            title="일별 PUR (결제 전환율)",
            labels={"pur": "전환율", "event_date": ""},
        )
        fig.update_yaxes(tickformat=".2%")
        fig.update_traces(line_color="#FF6B6B")
        st.plotly_chart(fig, use_container_width=True)

    # ── 코호트 리텐션 히트맵 ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📅 코호트별 리텐션 (Day-1 / Day-7 / Day-30)")
    st.caption("같은 달에 처음 접속한 유저들이 이후에 얼마나 남아있는지 추적합니다.")

    cohort_df = bq(f"""
        WITH first_activity AS (
            SELECT
                user_id,
                MIN(event_date)                        AS first_date,
                DATE_TRUNC(MIN(event_date), MONTH)     AS cohort_month
            FROM `{P}.events`
            GROUP BY user_id
        ),
        retention AS (
            SELECT
                f.cohort_month,
                COUNT(DISTINCT f.user_id) AS cohort_size,
                COUNT(DISTINCT CASE WHEN DATE_DIFF(e.event_date, f.first_date, DAY)
                    BETWEEN 1 AND 2  THEN f.user_id END) AS day1,
                COUNT(DISTINCT CASE WHEN DATE_DIFF(e.event_date, f.first_date, DAY)
                    BETWEEN 6 AND 8  THEN f.user_id END) AS day7,
                COUNT(DISTINCT CASE WHEN DATE_DIFF(e.event_date, f.first_date, DAY)
                    BETWEEN 28 AND 32 THEN f.user_id END) AS day30
            FROM first_activity f
            LEFT JOIN `{P}.events` e ON f.user_id = e.user_id
            GROUP BY f.cohort_month
        )
        SELECT
            FORMAT_DATE('%Y-%m', cohort_month) AS cohort,
            cohort_size,
            ROUND(SAFE_DIVIDE(day1,  cohort_size) * 100, 1) AS day1_pct,
            ROUND(SAFE_DIVIDE(day7,  cohort_size) * 100, 1) AS day7_pct,
            ROUND(SAFE_DIVIDE(day30, cohort_size) * 100, 1) AS day30_pct
        FROM retention
        ORDER BY cohort_month
    """)

    if not cohort_df.empty:
        import plotly.graph_objects as go_

        heat_data = cohort_df[["day1_pct", "day7_pct", "day30_pct"]].values.tolist()
        fig = go_.Figure(go_.Heatmap(
            z=heat_data,
            x=["Day-1", "Day-7", "Day-30"],
            y=cohort_df["cohort"].tolist(),
            colorscale="RdYlGn",
            zmin=0, zmax=100,
            text=[[f"{v}%" for v in row] for row in heat_data],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="리텐션율(%)"),
        ))
        fig.update_layout(title="코호트별 리텐션율 (%)", xaxis_title="", yaxis_title="가입 월")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            cohort_df.rename(columns={
                "cohort":      "가입월",
                "cohort_size": "코호트 유저수",
                "day1_pct":    "Day-1 (%)",
                "day7_pct":    "Day-7 (%)",
                "day30_pct":   "Day-30 (%)",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  AI 에이전트
# ══════════════════════════════════════════════════════════════════════════════
_SCHEMA = f"""
BigQuery 프로젝트: {P}

사용 가능한 테이블:

1. daily_game_metrics
   - metric_date DATE, DAU INT, MAU INT, DNU INT
   - total_revenue FLOAT, ARPU FLOAT, ARPPU FLOAT, PUR FLOAT
   - PCU INT, ACU INT

2. events  (유저×날짜 행동 집계)
   - user_id STRING, event_date DATE
   - login_yn BOOL, playtime_minutes INT, level INT
   - stage_clear_yn BOOL
   - gacha_result STRING (N/R/SR/SSR), gacha_try_count INT
   - item_use_count INT, session_count INT, consecutive_login_days INT

3. purchases  (결제 로그)
   - user_id STRING, purchase_date DATE
   - purchase_amount FLOAT  -- 990/1900/4900/9900/29900/49900/99900 (원)
   - item_category STRING  -- gacha/costume/stamina/package
   - is_first_purchase BOOL, payment_count INT, cumulative_amount FLOAT

4. marketing_events  (A/B 테스트 캠페인)
   - user_id STRING, campaign_date DATE
   - impression_yn BOOL, click_yn BOOL, conversion_yn BOOL
   - reward_amount FLOAT  -- 1000=A그룹, 1500=B그룹
   - ad_cost FLOAT, ad_revenue FLOAT

5. user_daily_snapshot  (유저 일별 스냅샷 + 모델 예측)
   - user_id STRING, snapshot_date DATE
   - churn_risk_score FLOAT  -- 0~1 (이탈 위험도)
   - max_level INT, total_playtime INT, total_gacha INT
   - total_spent FLOAT, active_days INT

규칙:
- 반드시 정확한 테이블 풀네임을 사용하세요: `{P}.테이블명`
- 날짜 비교는 DATE 함수 또는 CURRENT_DATE() 사용
- 결과는 최대 500행으로 LIMIT 제한
- A그룹: reward_amount = 1000, B그룹: reward_amount = 1500
"""

_SYSTEM_PROMPT = f"""당신은 Chronicle Tactics 모바일 게임의 데이터 분석 전문가입니다.
유저가 자연어로 질문하면 BigQuery SQL을 생성하여 데이터를 조회하고 마케팅/비즈니스 인사이트를 제공합니다.

{_SCHEMA}

응답 방식:
1. 어떤 쿼리를 실행할지 간단히 설명
2. run_bigquery_query 도구로 SQL 실행
3. 결과를 바탕으로 한국어로 명확한 인사이트 제공
4. 필요시 추가 쿼리로 심층 분석

항상 구체적인 숫자와 비율을 언급하고, 마케팅/비즈니스 관점의 액션 아이템을 제안하세요."""


def _run_agent(user_question: str) -> list[dict]:
    """Gemini Function Calling 기반 에이전트 실행. 메시지 리스트 반환."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return [{"type": "error", "text": "GEMINI_API_KEY가 설정되지 않았습니다. .env 파일에 추가해주세요."}]

    try:
        from google import genai as _genai
        from google.genai import types as _gtypes
    except ImportError:
        return [{"type": "error", "text": "google-genai 패키지가 설치되지 않았습니다. requirements.txt를 확인해주세요."}]

    client = _genai.Client(api_key=api_key)

    bq_tool = _gtypes.Tool(
        function_declarations=[
            _gtypes.FunctionDeclaration(
                name="run_bigquery_query",
                description="BigQuery SQL 쿼리를 실행하고 결과를 반환합니다. LIMIT 500 포함.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "sql":         {"type": "STRING", "description": "실행할 BigQuery SQL 쿼리"},
                        "description": {"type": "STRING", "description": "이 쿼리가 무엇을 분석하는지 한 줄 설명"},
                    },
                    "required": ["sql", "description"],
                },
            )
        ]
    )

    cfg = _gtypes.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        tools=[bq_tool],
    )

    contents = [_gtypes.Content(role="user", parts=[_gtypes.Part(text=user_question)])]
    output_blocks: list[dict] = []

    for _ in range(5):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=cfg,
        )

        candidate = response.candidates[0]
        fn_response_parts = []

        for part in candidate.content.parts:
            if getattr(part, "text", None):
                output_blocks.append({"type": "text", "text": part.text})

            fn_call = getattr(part, "function_call", None)
            if fn_call and fn_call.name == "run_bigquery_query":
                sql  = fn_call.args.get("sql", "")
                desc = fn_call.args.get("description", "")
                output_blocks.append({"type": "sql", "sql": sql, "desc": desc})

                try:
                    result_df = bq(sql)
                    result_str = result_df.to_string(index=False) if not result_df.empty else "결과 없음 (0행)"
                    if not result_df.empty:
                        output_blocks.append({"type": "table", "df": result_df})
                except Exception as e:
                    result_str = f"쿼리 오류: {e}"
                    output_blocks.append({"type": "error", "text": result_str})

                fn_response_parts.append(
                    _gtypes.Part(
                        function_response=_gtypes.FunctionResponse(
                            name=fn_call.name,
                            response={"result": result_str},
                        )
                    )
                )

        if not fn_response_parts:
            break

        contents.append(candidate.content)
        contents.append(_gtypes.Content(role="user", parts=fn_response_parts))

    return output_blocks


with tab5:
    st.subheader("🤖 AI 데이터 분석 에이전트")
    st.caption(
        "자연어로 질문하면 Gemini AI가 BigQuery SQL을 생성하고 게임 데이터를 분석합니다. "
        "(Google Gemini 1.5 Flash 기반 Function Calling)"
    )

    api_configured = bool(os.getenv("GEMINI_API_KEY", ""))
    if not api_configured:
        st.warning(
            "⚠️ **GEMINI_API_KEY**가 설정되지 않았습니다.  \n"
            "`.env` 파일에 `GEMINI_API_KEY=AIza...` 를 추가하고 컨테이너를 재시작하세요.  \n"
            "[Google AI Studio](https://aistudio.google.com/app/apikey) 에서 무료로 발급받을 수 있습니다."
        )

    # 예시 질문 버튼
    st.markdown("**예시 질문:**")
    ex_cols = st.columns(3)
    example_questions = [
        "A/B 테스트 결과를 요약해줘",
        "최근 30일 매출 트렌드 분석해줘",
        "이탈 위험 유저 상위 10명 보여줘",
        "카테고리별 결제 현황 알려줘",
        "Day-1 리텐션이 가장 높은 달은?",
        "ARPU가 가장 높은 날은 언제야?",
    ]
    for i, q in enumerate(example_questions):
        if ex_cols[i % 3].button(q, key=f"ex_{i}", disabled=not api_configured):
            st.session_state["ai_prefill"] = q

    st.markdown("---")

    # 채팅 히스토리 초기화
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []
    if "ai_prefill" not in st.session_state:
        st.session_state["ai_prefill"] = ""

    # 히스토리 렌더링
    for turn in st.session_state["ai_chat_history"]:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.write(turn["content"])
            else:
                for block in turn["blocks"]:
                    if block["type"] == "text":
                        st.write(block["text"])
                    elif block["type"] == "sql":
                        with st.expander(f"🔍 SQL: {block['desc']}", expanded=False):
                            st.code(block["sql"], language="sql")
                    elif block["type"] == "table":
                        st.dataframe(block["df"], use_container_width=True, hide_index=True)
                    elif block["type"] == "error":
                        st.error(block["text"])

    # 채팅 입력
    prefill = st.session_state.pop("ai_prefill", "")
    user_input = st.chat_input(
        "질문을 입력하세요. 예: A/B 테스트 CVR 차이가 통계적으로 유의한가?",
        disabled=not api_configured,
        key="ai_chat_input",
    )
    if prefill:
        user_input = prefill

    if user_input:
        st.session_state["ai_chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Gemini가 분석 중입니다..."):
                blocks = _run_agent(user_input)

            for block in blocks:
                if block["type"] == "text":
                    st.write(block["text"])
                elif block["type"] == "sql":
                    with st.expander(f"🔍 SQL: {block['desc']}", expanded=False):
                        st.code(block["sql"], language="sql")
                elif block["type"] == "table":
                    st.dataframe(block["df"], use_container_width=True, hide_index=True)
                elif block["type"] == "error":
                    st.error(block["text"])

        st.session_state["ai_chat_history"].append({"role": "assistant", "blocks": blocks})

    if st.session_state["ai_chat_history"]:
        if st.button("🗑️ 대화 초기화", key="clear_chat"):
            st.session_state["ai_chat_history"] = []
            st.rerun()
