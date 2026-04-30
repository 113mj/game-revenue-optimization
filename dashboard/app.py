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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🧪 A/B Test",
    "🔮 Churn / LTV",
    "🔽 Funnel & Retention",
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
