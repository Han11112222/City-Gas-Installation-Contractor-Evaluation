from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="도시가스 신규계량기 사용량 기반 우수 시공업체 평가",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "20251204-수요개발_신규계량기사용량현황.xlsx"

# 단독주택 월별 평균사용량 (2024년 기준, 부피 m³)
SINGLE_DETACHED_MONTHLY_AVG = {
    1: 96,
    2: 92,
    3: 67,
    4: 41,
    5: 25,
    6: 16,
    7: 9,
    8: 8,
    9: 7,
    10: 9,
    11: 21,
    12: 55,
}

# 포상 기준 (연간 10전 이상, 연간 10만 m³ 이상)
MIN_METERS = 10        # 연간 10전 이상
MIN_ANNUAL = 100_000   # 연간 100,000 m³ 이상


# --------------------------------------------------
# 유틸 함수
# --------------------------------------------------
def fmt_int(x: float) -> str:
    """정수 + 천단위 콤마"""
    return f"{int(round(x)):,}"


def get_month_cols(df: pd.DataFrame):
    """연월(YYYYMM) 숫자형 컬럼만 추출"""
    return [c for c in df.columns if isinstance(c, (int, np.integer))]


def build_detached_avg_by_col(month_cols):
    """연월 컬럼명에 단독주택 월평균 사용량 매핑"""
    mapping = {}
    for col in month_cols:
        month_num = int(str(col)[-2:])  # 202501 -> 1, 202412 -> 12
        mapping[col] = SINGLE_DETACHED_MONTHLY_AVG.get(month_num, np.nan)
    return mapping


def center_style(df: pd.DataFrame, highlight_fn=None):
    """
    모든 셀/헤더 가로 중앙정렬 + (옵션) 행 단위 하이라이트.
    highlight_fn(row) -> CSS 문자열 리스트
    """
    styler = df.style
    if highlight_fn is not None:
        styler = styler.apply(highlight_fn, axis=1)

    # 전체 숫자/텍스트 중앙정렬
    styler = styler.set_properties(**{"text-align": "center"})
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("text-align", "center")],
            }
        ]
    )
    return styler


# --------------------------------------------------
# 데이터 불러오기 & 전처리
# --------------------------------------------------
@st.cache_data
def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def preprocess(df_raw: pd.DataFrame):
    """
    사용 예정량 산정 로직

      1) 업종: 가스시공업 제1종만 사용
      2) 자체업종명: 아파트 제외
      3) 연립/다세대 → 용도 '단독주택' 으로 변경
      4) 가정용(단독주택):
         - 월 사용량이 NaN 또는 0이면 단독주택 월평균표로 강제 치환
         - 치환된 1~12월을 그대로 합산 → 연간사용량_추정
      5) 가정용 외:
         - 용도에서 단독주택을 제외한 나머지
         - 월별 값 중 숫자가 있는 달만 골라 평균(= 합계 / 값이 있는 달 수)
         - 월평균 × 12개월 → 연간사용량_추정
    """
    df = df_raw.copy()

    # 1종 시공업체만 사용
    df = df[df["업종"] == "가스시공업 제1종"].copy()

    month_cols = get_month_cols(df)
    detached_avg_by_col = build_detached_avg_by_col(month_cols)

    # 아파트 제외
    df = df[df["자체업종명"] != "아파트"].copy()

    # 연립/다세대 -> 단독주택
    mask_multi = df["자체업종명"].isin(["연립주택", "다세대주택"])
    df.loc[mask_multi, "용도"] = "단독주택"

    # 사용여부 'Y' 만 사용 (있으면 적용)
    if "사용여부" in df.columns:
        df = df[df["사용여부"] == "Y"].copy()

    # 계량기별 연간 사용량 추정
    def compute_annual(row):
        usage = row[month_cols].astype(float)

        # ── 가정용: 단독주택 ─────────────────────
        if row["용도"] == "단독주택":
            for col in month_cols:
                base = detached_avg_by_col.get(col)
                v = usage[col]
                # 빈칸(NaN) 또는 0 → 단독주택 월평균으로 강제 치환
                if pd.isna(v) or v == 0:
                    if not pd.isna(base):
                        usage[col] = base
            return float(usage.sum())

        # ── 가정용 외: 단독주택 제외 나머지 ───────
        else:
            # 값이 있는 달만 사용(블랭크만 제외, 0은 그대로 둠)
            vals = usage.dropna()
            if len(vals) == 0:
                return 0.0
            monthly_avg = float(vals.mean())  # 예: 3달 값 있으면 /3
            return monthly_avg * 12.0        # 월평균 × 12개월

    df["연간사용량_추정"] = df.apply(compute_annual, axis=1)

    # 대분류(설명용): 가정용 vs 가정용외
    df["대분류"] = np.where(df["용도"] == "단독주택", "가정용(단독주택)", "가정용외")

    # 시공업체별 집계 (전체 기준)
    agg = (
        df.groupby("시공업체", as_index=True)
        .agg(
            신규계량기수=("계량기번호", "nunique"),
            연간사용량합계=("연간사용량_추정", "sum"),
        )
    )
    agg["계량기당_평균연간사용량"] = agg["연간사용량합계"] / agg["신규계량기수"]

    # 포상 기준 충족 업체 (10전 이상 + 연간 10만 m³ 이상)
    eligible = agg[
        (agg["신규계량기수"] >= MIN_METERS)
        & (agg["연간사용량합계"] >= MIN_ANNUAL)
    ].copy()
    eligible = eligible.sort_values("연간사용량합계", ascending=False)
    eligible["순위"] = np.arange(1, len(eligible) + 1)

    # 업체 × 용도별 사용량 + 전수 (전체)
    usage_by_type = (
        df.groupby(["시공업체", "용도"])
        .agg(
            연간사용량_추정=("연간사용량_추정", "sum"),
            전수=("계량기번호", "nunique"),
        )
        .reset_index()
    )

    # 가정용외 집계: 단독주택·공동주택 모두 제외한 나머지 용도
    df_nonres_for_type = df[
        (df["용도"] != "단독주택") & (df["용도"] != "공동주택")
    ].copy()
    usage_by_type_nonres = (
        df_nonres_for_type.groupby(["시공업체", "용도"])
        .agg(
            연간사용량_추정=("연간사용량_추정", "sum"),
            전수=("계량기번호", "nunique"),
        )
        .reset_index()
    )

    return df, agg, eligible, usage_by_type, usage_by_type_nonres, month_cols


# --------------------------------------------------
# 메인
# --------------------------------------------------
st.title("도시가스 신규계량기 사용량 기반 우수 시공업체 평가")

st.markdown(
    """
- **대상 데이터** : 수요개발 신규계량기 사용량 현황(엑셀)
- **분석 대상 시공업체** : 가스시공업 **제1종** 시공업체
- **포상 기본 전제**
  - 연간 신규계량기 수 **10전 이상**
  - 추정 연간사용량 합계 **100,000 m³ 이상**
"""
)

# 파일 업로드 (없으면 저장소 내 기본 파일 사용)
uploaded = st.file_uploader("엑셀 파일 업로드 (없으면 기본 파일 사용)", type=["xlsx"])
if uploaded is not None:
    raw_df = pd.read_excel(uploaded)
else:
    raw_df = load_raw(DATA_FILE)

# 아파트 포함 전체 신규계량기수(1종, 사용여부 Y만) 계산
total_meters_incl_apt = None
if {"업종", "계량기번호"}.issubset(raw_df.columns):
    df_meter = raw_df[raw_df["업종"] == "가스시공업 제1종"].copy()
    if "사용여부" in df_meter.columns:
        df_meter = df_meter[df_meter["사용여부"] == "Y"].copy()
    total_meters_incl_apt = df_meter["계량기번호"].nunique()

(
    df_proc,
    agg_all,
    eligible,
    usage_by_type,
    usage_by_type_nonres,
    month_cols,
) = preprocess(raw_df)

# 전체 사용량 & 상위 10개 집중도
total_usage_all = agg_all["연간사용량합계"].sum()
all_rank_for_share = agg_all.sort_values("연간사용량합계", ascending=False)
top10_usage = all_rank_for_share["연간사용량합계"].head(10).sum()
top10_share = top10_usage / total_usage_all if total_usage_all > 0 else 0.0

# 상단 KPI
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("전체 시공업체 수 (1종)", f"{agg_all.shape[0]:,} 개")
with col2:
    st.metric("포상 기준 충족 업체 수", f"{eligible.shape[0]:,} 개")
with col3:
    st.metric(
        "전체 신규계량기 수 (아파트 제외)",
        f"{df_proc['계량기번호'].nunique():,} 전",
    )
with col4:
    if total_meters_incl_apt is not None:
        st.metric(
            "전체 신규계량기 수 (아파트 포함)",
            f"{total_meters_incl_apt:,} 전",
        )

tab_rank, tab_type, tab_detail, tab_final = st.tabs(
    ["업체별 순위", "용도별 분석", "업체별 용도 분석", "최종분석"]
)

# --------------------------------------------------
# 탭 1 : 업체별 순위
# --------------------------------------------------
with tab_rank:
    st.subheader("📈 포상 기준 + 전체 업체 순위 (연간 사용량 기준)")

    # 전체 업체 순위 (연간 사용량 기준)
    all_rank = agg_all.sort_values("연간사용량합계", ascending=False).reset_index()
    all_rank["순위"] = np.arange(1, len(all_rank) + 1)
    all_rank["신규계량기 수(전)"] = all_rank["신규계량기수"]
    all_rank["추정 연간사용량 합계(m³)"] = all_rank["연간사용량합계"].map(fmt_int)

    disp_cols_all = [
        "순위",
        "시공업체",
        "신규계량기 수(전)",
        "추정 연간사용량 합계(m³)",
    ]

    def highlight_eligible(row):
        cond = row["시공업체"] in eligible.index
        return ["background-color: #FFF4CC" if cond else "" for _ in row]

    styled_all_rank = center_style(all_rank[disp_cols_all], highlight_eligible)

    st.dataframe(
        styled_all_rank,
        use_container_width=True,
        hide_index=True,
        column_config={
            # 순위 컬럼 폭을 줄이고 중앙정렬 유지 (NumberColumn → Column)
            "순위": st.column_config.Column("순위", width="small"),
        },
    )

    st.caption(
        "- 노란색으로 표시된 행이 포상 기준(10전 이상 & 100,000 m³ 이상)을 충족하는 시공업체.\n"
        f"- 전체 1종 시공업체의 추정 연간사용량 합계는 **{fmt_int(total_usage_all)} m³** 이며,\n"
        f"  이 중 상위 10개 업체 비중은 약 **{top10_share * 100:,.1f}%**."
    )

    # 포상 기준 충족 업체만 별도 차트
    st.markdown("---")
    st.markdown("#### 🏆 포상 기준 충족 업체 상위 사용량")

    if eligible.empty:
        st.info("포상 기준(10전 이상 & 연간 100,000 m³ 이상)을 만족하는 업체가 없습니다.")
    else:
        rank_df = (
            eligible.reset_index()
            .sort_values("연간사용량합계", ascending=False)
            .copy()
        )
        rank_df["시공업체명"] = rank_df["시공업체"]
        rank_df["연간총"] = rank_df["연간사용량합계"]
        rank_df["추정 연간사용량 합계(m³)"] = rank_df["연간총"].map(fmt_int)

        chart_df = rank_df.head(min(20, rank_df.shape[0]))
        fig = px.bar(
            chart_df,
            x="시공업체명",
            y="연간총",
            text="추정 연간사용량 합계(m³)",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="시공업체",
            yaxis_title="추정 연간사용량 합계(m³)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 탭 2 : 용도별 분석 (가정용 vs 가정용외)
# --------------------------------------------------
with tab_type:
    st.subheader("📊 대분류별 사용량 요약 (가정용 vs 가정용외)")

    # 가정용(단독주택) / 가정용외(단독·공동 제외)
    df_home = df_proc[df_proc["용도"] == "단독주택"].copy()
    df_nonres_rows = df_proc[
        (df_proc["용도"] != "단독주택") & (df_proc["용도"] != "공동주택")
    ].copy()

    total_m3 = df_proc["연간사용량_추정"].sum()

    rows = [
        {
            "대분류": "가정용(공동주택 제외)",
            "계량기 수(전)": df_home["계량기번호"].nunique(),
            "추정 연간사용량(m³)": df_home["연간사용량_추정"].sum(),
        },
        {
            "대분류": "가정용외",
            "계량기 수(전)": df_nonres_rows["계량기번호"].nunique(),
            "추정 연간사용량(m³)": df_nonres_rows["연간사용량_추정"].sum(),
        },
        {
            "대분류": "합계",
            "계량기 수(전)": df_proc["계량기번호"].nunique(),
            "추정 연간사용량(m³)": total_m3,
        },
    ]
    big_df = pd.DataFrame(rows)

    # 비중 계산
    big_df["사용량 비중(%)"] = (
        big_df["추정 연간사용량(m³)"] / total_m3 * 100 if total_m3 > 0 else 0
    )
    big_df.loc[big_df["대분류"] == "합계", "사용량 비중(%)"] = 100.0

    big_df["계량기 수(전)"] = big_df["계량기 수(전)"].map(lambda x: f"{int(x):,}")
    big_df["추정 연간사용량(m³)"] = big_df["추정 연간사용량(m³)"].map(fmt_int)
    big_df["사용량 비중(%)"] = big_df["사용량 비중(%)"].map(
        lambda x: f"{x:,.1f}%" if x != 0 else "0.0%"
    )

    styled_big = center_style(big_df)

    st.dataframe(
        styled_big,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### 📌 대분류별·용도별 시공업체 순위")

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(
        ["가정용(단독주택) 순위", "가정용외 순위", "가정용외 용도별 분석"]
    )

    # ── 가정용(단독주택) 순위 ───────────────────────────
    with sub_tab1:
        res = usage_by_type[usage_by_type["용도"] == "단독주택"].copy()
        if res.empty:
            st.info("단독주택 데이터가 없습니다.")
        else:
            res = res.sort_values("연간사용량_추정", ascending=False)
            res["순위"] = np.arange(1, len(res) + 1)
            res["연간총"] = res["연간사용량_추정"]
            res["추정 연간사용량(m³)"] = res["연간총"].map(fmt_int)
            res["전수(전)"] = res["전수"].map(lambda x: f"{int(x):,}")

            disp = res[["순위", "시공업체", "추정 연간사용량(m³)", "전수(전)"]]
            styled_res = center_style(disp)

            st.dataframe(
                styled_res,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "순위": st.column_config.Column("순위", width="small"),
                },
            )

            top_n = min(15, res.shape[0])
            chart_res = res.head(top_n)
            fig_res = px.bar(
                chart_res,
                x="시공업체",
                y="연간총",
                text="추정 연간사용량(m³)",
            )
            fig_res.update_traces(textposition="outside")
            fig_res.update_layout(
                xaxis_title="시공업체",
                yaxis_title="단독주택 추정 연간사용량(m³)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_res, use_container_width=True)

    # ── 가정용외 전체 순위 ─────────────────────────────
    with sub_tab2:
        nonres_comp = (
            usage_by_type_nonres.groupby("시공업체")
            .agg(
                연간사용량_추정=("연간사용량_추정", "sum"),
                전수=("전수", "sum"),
            )
            .reset_index()
        )
        if nonres_comp.empty:
            st.info("가정용외 데이터가 없습니다.")
        else:
            nonres_comp = nonres_comp.sort_values(
                "연간사용량_추정", ascending=False
            )
            nonres_comp["순위"] = np.arange(1, len(nonres_comp) + 1)
            nonres_comp["연간총"] = nonres_comp["연간사용량_추정"]
            nonres_comp["추정 연간사용량(m³)"] = nonres_comp["연간총"].map(fmt_int)
            nonres_comp["전수(전)"] = nonres_comp["전수"].map(
                lambda x: f"{int(x):,}"
            )

            disp = nonres_comp[
                ["순위", "시공업체", "추정 연간사용량(m³)", "전수(전)"]
            ]
            styled_nonres_comp = center_style(disp)

            st.dataframe(
                styled_nonres_comp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "순위": st.column_config.Column("순위", width="small"),
                },
            )

            top_n2 = min(15, nonres_comp.shape[0])
            chart_nonres = nonres_comp.head(top_n2)
            fig_nonres = px.bar(
                chart_nonres,
                x="시공업체",
                y="연간총",
                text="추정 연간사용량(m³)",
            )
            fig_nonres.update_traces(textposition="outside")
            fig_nonres.update_layout(
                xaxis_title="시공업체",
                yaxis_title="가정용외 추정 연간사용량(m³)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_nonres, use_container_width=True)

    # ── 가정용외 용도별 분석 ───────────────────────────
    with sub_tab3:
        st.markdown("##### 📌 가정용외 용도별 1위 시공업체")

        type_summary_nonres = (
            usage_by_type_nonres.groupby("용도")
            .agg(
                총연간사용량=("연간사용량_추정", "sum"),
                업체수=("시공업체", "nunique"),
            )
            .reset_index()
        )
        idx = usage_by_type_nonres.groupby("용도")["연간사용량_추정"].idxmax()
        top_per_type_nonres = usage_by_type_nonres.loc[
            idx, ["용도", "시공업체", "연간사용량_추정", "전수"]
        ]
        type_summary_nonres = type_summary_nonres.merge(
            top_per_type_nonres, on="용도", how="left"
        )

        if type_summary_nonres.empty:
            st.info("가정용외 용도 데이터가 없습니다.")
        else:
            type_disp = type_summary_nonres.copy()
            type_disp["1위 연간사용량(m³)"] = type_disp[
                "연간사용량_추정"
            ].map(fmt_int)
            type_disp["1위 전수(전)"] = type_disp["전수"].map(
                lambda x: f"{int(x):,}"
            )
            type_disp = type_disp.rename(
                columns={
                    "시공업체": "1위 시공업체",
                }
            )

            disp = type_disp[
                ["용도", "1위 시공업체", "1위 연간사용량(m³)", "1위 전수(전)"]
            ]
            styled_type_summary = center_style(disp)

            st.dataframe(
                styled_type_summary,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.markdown("##### 📌 가정용외 용도별 시공업체 순위")

            type_list_nonres = sorted(type_disp["용도"].unique().tolist())
            selected_type = st.selectbox(
                "용도 선택 (가정용외)", type_list_nonres
            )

            sub = usage_by_type_nonres[
                usage_by_type_nonres["용도"] == selected_type
            ].copy()
            sub = sub.sort_values("연간사용량_추정", ascending=False)
            sub["순위"] = np.arange(1, len(sub) + 1)
            sub["연간총"] = sub["연간사용량_추정"]
            sub["추정 연간사용량(m³)"] = sub["연간총"].map(fmt_int)
            sub["전수(전)"] = sub["전수"].map(lambda x: f"{int(x):,}")

            disp_rank = sub[
                ["순위", "시공업체", "추정 연간사용량(m³)", "전수(전)"]
            ]
            styled_sub = center_style(disp_rank)

            st.dataframe(
                styled_sub,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "순위": st.column_config.Column("순위", width="small"),
                },
            )

            top_n_type = min(15, sub.shape[0])
            chart_type = sub.head(top_n_type)
            fig_type = px.bar(
                chart_type,
                x="시공업체",
                y="연간총",
                text="추정 연간사용량(m³)",
            )
            fig_type.update_traces(textposition="outside")
            fig_type.update_layout(
                xaxis_title="시공업체",
                yaxis_title=f"{selected_type} 추정 연간사용량(m³)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_type, use_container_width=True)

            # ── 선택 용도별 상세 리스트 (계량기별 시공 내역) ─────────
            st.markdown("---")
            st.markdown(f"##### 🧾 {selected_type} 상세 리스트 (시공업체별 시공 내역)")

            company_list = sub["시공업체"].tolist()
            selected_company_type = st.selectbox(
                f"{selected_type} 시공업체 선택", company_list
            )

            detail = df_proc[
                (df_proc["용도"] == selected_type)
                & (df_proc["시공업체"] == selected_company_type)
            ].copy()

            if detail.empty:
                st.info("선택한 시공업체의 해당 용도 시공 내역이 없습니다.")
            else:
                detail = detail.sort_values("연간사용량_추정", ascending=False)
                detail["연간사용량_추정(m³)"] = detail["연간사용량_추정"].map(
                    fmt_int
                )
                detail_cols = [
                    "계량기번호",
                    "고객명",
                    "주소",
                    "자체업종명",
                    "연간사용량_추정(m³)",
                ]
                exist_cols = [c for c in detail_cols if c in detail.columns]

                styled_detail = center_style(detail[exist_cols])

                st.dataframe(
                    styled_detail,
                    use_container_width=True,
                    hide_index=True,
                )


# --------------------------------------------------
# 탭 3 : 업체별 용도 분석
# --------------------------------------------------
with tab_detail:
    st.subheader("📌 업체별 용도별 사용 패턴")

    if eligible.empty:
        st.info("포상 기준을 만족하는 업체가 없어서 상세 분석 대상이 없습니다.")
    else:
        target_companies = eligible.index.tolist()
        selected_company = st.selectbox(
            "시공업체 선택 (포상 기준 충족 업체 기준)",
            target_companies,
            index=0,
        )

        comp_df = usage_by_type[
            usage_by_type["시공업체"] == selected_company
        ].copy()
        comp_df = comp_df.sort_values("연간사용량_추정", ascending=False)
        comp_df["연간총"] = comp_df["연간사용량_추정"]
        comp_df["추정 연간사용량(m³)"] = comp_df["연간총"].map(fmt_int)
        comp_df["전수(전)"] = comp_df["전수"].map(lambda x: f"{int(x):,}")

        st.markdown(f"**선택한 시공업체 : {selected_company}**")

        fig2 = px.bar(
            comp_df,
            x="용도",
            y="연간총",
            text="추정 연간사용량(m³)",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            xaxis_title="용도",
            yaxis_title="추정 연간사용량(m³)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        disp_comp = comp_df[["용도", "추정 연간사용량(m³)", "전수(전)"]]
        styled_comp = center_style(disp_comp)

        st.dataframe(
            styled_comp,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# 탭 4 : 최종분석 – 종합점수 + 2-3항목 기반 포상 추천
# --------------------------------------------------
with tab_final:
    st.subheader("※ 최종분석 – 종합점수 + 2-3항목(기존주택 비율) 기반 포상 추천")

    st.markdown(
        """
- **별도의 파일 업로드는 필요 없음.**  
  위에서 선택한 **신규계량기 사용량 엑셀 파일** 안에서  
  시공업체 **평가점수표(1-1~3-2, 감점, 총점 포함)** 시트를 자동으로 찾습니다.
- 평가표 시트에서 **총점 + 2-3항목(기존주택 개발 비율)** 점수를 활용해  
  본상/특별상 후보를 자동 추천합니다.
- 기존주택 비율은 이미 평가기준 **2-3항목 점수**로 반영되어 있으므로,  
  이 탭에서는 비율을 다시 계산하지 않고 **2-3 점수 컬럼**을 그대로 사용합니다.
"""
    )

    # 어떤 통합문서를 볼지 선택 (위에서 이미 업로드했던 것 재사용)
    if uploaded is not None:
        excel_target = uploaded
    else:
        excel_target = DATA_FILE  # 기본 파일 사용

    # 통합문서 안에서 '구분' + '총점' 헤더가 있는 시트를 자동 탐색
    xls = pd.ExcelFile(excel_target)
    df_eval = None
    found_sheet_name = None
    header_row_idx = None

    for sheet in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=sheet, header=None)
        for i, row in tmp.iterrows():
            vals = [str(v).strip() for v in row.values if isinstance(v, str)]
            if ("구분" in vals) and ("총점" in vals):
                df_eval = pd.read_excel(xls, sheet_name=sheet, header=i)
                header_row_idx = i
                found_sheet_name = sheet
                break
        if df_eval is not None:
            break

    if df_eval is None:
        st.error(
            "현재 엑셀 파일 안에서 '구분'과 '총점' 헤더를 가진 평가점수표 시트를 찾지 못했습니다. "
            "평가표 시트가 같은 파일 안에 있는지, 그리고 헤더 이름이 '구분', '총점'으로 되어 있는지 확인해줘."
        )
    else:
        st.markdown(f"✅ 사용된 평가점수표 시트: **`{found_sheet_name}`** (헤더 행: {header_row_idx+1}행)")

        df_eval.columns = [str(c).strip() for c in df_eval.columns]

        required_cols = ["구분", "2-3", "총점"]
        missing = [c for c in required_cols if c not in df_eval.columns]
        if missing:
            st.error(
                f"평가점수표에서 다음 컬럼을 찾지 못했어: {', '.join(missing)}  \n"
                "시트 안 헤더 이름이 그림의 '구분', '2-3', '총점'과 동일한지 확인해줘."
            )
        else:
            # 합계 같은 행 제거
            df_eval = df_eval[~df_eval["구분"].isna()].copy()
            df_eval = df_eval[
                ~df_eval["구분"].astype(str).str.contains("합계")
            ].copy()

            # 점수형 컬럼 숫자로 변환
            score_cols = ["1-1", "2-1", "2-2", "2-3", "3-1", "3-2", "감점", "총점"]
            for col in score_cols:
                if col in df_eval.columns:
                    df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

            # 총점 기준 정렬 및 순위
            df_eval = df_eval.sort_values("총점", ascending=False).reset_index(drop=True)
            df_eval["순위(총점기준)"] = np.arange(1, len(df_eval) + 1)

            # 본상: 총점 1위
            main_row = df_eval.iloc[0]
            main_winner = str(main_row["구분"])

            # 특별상: 본상 제외 후 2-3 점수 → 총점 순
            cand = df_eval[df_eval["구분"] != main_winner].copy()
            special_winner = None
            special_row = None
            if not cand.empty:
                cand = cand.sort_values(["2-3", "총점"], ascending=[False, False])
                special_row = cand.iloc[0]
                special_winner = str(special_row["구분"])

            # 포상구분 표시
            def mark_award(row):
                name = str(row["구분"])
                if name == main_winner:
                    return "본상(종합 1위)"
                elif special_winner is not None and name == special_winner:
                    return "특별상(기존주택 비율 우수)"
                else:
                    return ""

            df_eval["포상구분"] = df_eval.apply(mark_award, axis=1)

            # 출력용 테이블
            disp_cols = [
                "순위(총점기준)",
                "구분",
                "1-1",
                "2-1",
                "2-2",
                "2-3",
                "3-1",
                "3-2",
                "감점",
                "총점",
                "포상구분",
            ]
            exist_cols = [c for c in disp_cols if c in df_eval.columns]
            view_eval = df_eval[exist_cols].copy()

            for col in score_cols:
                if col in view_eval.columns:
                    view_eval[col] = view_eval[col].map(
                        lambda x: f"{int(x)}" if pd.notna(x) else ""
                    )
            if "순위(총점기준)" in view_eval.columns:
                view_eval["순위(총점기준)"] = view_eval["순위(총점기준)"].astype(int)

            def highlight_award(row):
                if row.get("포상구분") == "본상(종합 1위)":
                    color = "#FFF4CC"  # 본상: 연노랑
                elif str(row.get("포상구분", "")).startswith("특별상"):
                    color = "#E3F2FD"  # 특별상: 연하늘
                else:
                    color = ""
                return [f"background-color: {color}" for _ in row]

            styled_eval = center_style(view_eval, highlight_fn=highlight_award)

            st.markdown("#### 🧾 평가점수표 기반 최종 순위 및 포상 추천")
            st.dataframe(
                styled_eval,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "순위(총점기준)": st.column_config.Column("순위", width="small"),
                },
            )

            # 보고서용 요약 문장
            st.markdown("#### 📄 보고서용 요약 문장")

            main_total = int(main_row["총점"]) if not pd.isna(main_row["총점"]) else None
            main_23 = int(main_row["2-3"]) if not pd.isna(main_row["2-3"]) else None

            if special_row is not None:
                spec_total = int(special_row["총점"]) if not pd.isna(special_row["총점"]) else None
                spec_23 = int(special_row["2-3"]) if not pd.isna(special_row["2-3"]) else None
            else:
                spec_total = None
                spec_23 = None

            st.markdown(
                f"""
- **본상(우수 시공업체)** : `{main_winner}`  
  - 종합점수 **{main_total}점**으로 전체 1위를 기록하였으며,  
    가스시공업 허가취득 연수·공급전 세대수·연 사용예정량·품질관리(시공 부적합 비율, 준공서류 이관율) 등  
    전 항목에서 고르게 우수한 실적을 나타냅니다.
"""
            )

            if special_winner is not None:
                st.markdown(
                    f"""
- **특별상(기존주택 개발 우수)** : `{special_winner}`  
  - **2-3항목(기존 주택 개발 비율)** 점수 **{spec_23}점** 및 종합점수 **{spec_total}점**으로 상위권을 유지하고 있으며,  
    기존 가스배관이 구축된 지역 내 미공급 세대 발굴과 음식점·상가 등 고부가가치 수요개발에  
    두드러진 성과를 보인 업체로 평가됩니다.
"""
                )
