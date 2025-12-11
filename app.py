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
# 탭 4 : 최종 분석 (본상/특별상 추천)
# --------------------------------------------------
with tab_final:
    st.subheader("🏁 최종분석 – 종합점수 + 2-3항목(기존주택 비율) 기반 포상 추천")

    st.markdown(
        """
- 별도의 **시공업체 평가점수표**(1-1~3-2, 감점, 총점 포함)를 업로드하면  
  종합점수와 2-3항목(기존주택 개발 비율) 점수를 활용해 **본상/특별상 후보를 자동 추천**합니다.
- 기존주택 비율은 이미 **평가기준 2-3항목 점수**로 반영되어 있으므로,  
  이 탭에서는 별도 비율을 다시 계산하지 않고 **2-3 점수 컬럼**을 그대로 사용합니다.
"""
    )

    score_file = st.file_uploader(
        "시공업체 평가점수표 업로드 (2-3항목, 총점 포함)", type=["xlsx"], key="score_file"
    )

    if score_file is None:
        st.info("평가점수표 엑셀을 업로드하면 최종 분석 결과가 표시됩니다.")
    else:
        df_score = pd.read_excel(score_file)

        # 필수 키: 시공업체
        if "시공업체" not in df_score.columns:
            st.error("점수표에 '시공업체' 컬럼이 필요합니다.")
        else:
            # 총점 컬럼 / 2-3항목 컬럼 자동 탐색
            total_col = None
            exist_col = None
            for c in df_score.columns:
                name = str(c)
                if ("총점" in name) and total_col is None:
                    total_col = c
                if (("2-3" in name) or ("기존" in name)) and exist_col is None:
                    exist_col = c

            if total_col is None or exist_col is None:
                msg = []
                if total_col is None:
                    msg.append("· '총점' 이 포함된 컬럼 이름이 필요합니다.")
                if exist_col is None:
                    msg.append("· '2-3' 또는 '기존' 이 포함된 컬럼 이름이 필요합니다.")
                st.error("점수표 컬럼을 찾지 못했습니다.\n" + "\n".join(msg))
            else:
                # 기본 집계(신규계량기수, 연간사용량)와 merge
                base = agg_all.reset_index()[["시공업체", "신규계량기수", "연간사용량합계"]]
                merged = df_score.merge(base, on="시공업체", how="left")

                # 영업용 사용량 집계 (가정용외 중 용도명에 '영업' 포함하는 건)
                biz = (
                    usage_by_type_nonres[
                        usage_by_type_nonres["용도"].astype(str).str.contains("영업", na=False)
                    ]
                    .groupby("시공업체")
                    .agg(영업용_사용량=("연간사용량_추정", "sum"))
                    .reset_index()
                )
                merged = merged.merge(biz, on="시공업체", how="left")
                merged["영업용_사용량"] = merged["영업용_사용량"].fillna(0.0)

                # 영업용 비중(%)
                merged["영업용_비중(%)"] = np.where(
                    merged["연간사용량합계"] > 0,
                    merged["영업용_사용량"] / merged["연간사용량합계"] * 100,
                    0.0,
                )

                # 총점 기준 순위
                merged = merged.sort_values(total_col, ascending=False).reset_index(drop=True)
                merged["총점순위"] = np.arange(1, len(merged) + 1)

                # 요약표 표시
                summary_cols = [
                    "총점순위",
                    "시공업체",
                    total_col,
                    exist_col,
                    "신규계량기수",
                    "연간사용량합계",
                    "영업용_사용량",
                    "영업용_비중(%)",
                ]
                disp_summary = merged[summary_cols].copy()
                disp_summary = disp_summary.rename(
                    columns={
                        total_col: "총점",
                        exist_col: "2-3항목(기존주택 비율) 점수",
                        "연간사용량합계": "추정 연간사용량 합계(m³)",
                    }
                )

                disp_summary["신규계량기수"] = disp_summary["신규계량기수"].map(
                    lambda x: f"{int(x):,}"
                )
                disp_summary["추정 연간사용량 합계(m³)"] = disp_summary[
                    "추정 연간사용량 합계(m³)"
                ].map(fmt_int)
                disp_summary["영업용_사용량"] = disp_summary["영업용_사용량"].map(fmt_int)
                disp_summary["영업용_비중(%)"] = disp_summary["영업용_비중(%)"].map(
                    lambda v: f"{v:,.1f}%"
                )

                st.markdown("### 1) 최종 평가 요약표 (총점 + 2-3항목 + 사용실적)")
                st.dataframe(
                    center_style(disp_summary),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "총점순위": st.column_config.Column("총점순위", width="small"),
                    },
                )

                # 본상: 총점 1위
                main_row = merged.iloc[0]
                main_company = main_row["시공업체"]

                # 특별상: 영업용 사용량 기준 상위, 단 본상 업체는 제외
                biz_rank = merged.sort_values(
                    ["영업용_사용량", total_col], ascending=[False, False]
                )
                biz_rank = biz_rank[biz_rank["시공업체"] != main_company]
                if len(biz_rank) > 0:
                    special_row = biz_rank.iloc[0]
                    special_company = special_row["시공업체"]
                else:
                    special_row = None
                    special_company = None

                st.markdown("---")
                st.markdown("### 2) 포상 추천 결과")

                # 본상 블록
                st.markdown(f"#### 🏅 본상(우수 시공업체) 추천 : **{main_company}**")

                st.markdown(
                    f"""
- **총점** : {main_row[total_col]} 점  
- **2-3항목(기존주택 개발 비율) 점수** : {main_row[exist_col]} 점  
- **신규계량기 수** : {int(main_row["신규계량기수"]):,} 전  
- **추정 연간사용량 합계** : {fmt_int(main_row["연간사용량합계"])} m³  
- **영업용 사용량 비중** : {main_row["영업용_비중(%)"]:.1f}%  
"""
                )

                st.markdown(
                    """
**선정 논리 (보고서용 문장 예시)**  

1. 평가표 상 1-1, 2-1, 2-2, 2-3, 3-1, 3-2 전 항목을 합산한 **종합점수(총점) 1위**로,  
   경영·수요개발·품질 측면에서 전반적인 성과가 가장 우수함.
2. 특히 **2-3항목(기존주택 개발 비율)** 점수가 높아,  
   기존에 가스배관이 구축된 지역을 중심으로 **효율적인 관로 활용·수요개발**을 수행한 것으로 평가됨.
3. 신규계량기 수와 연간사용량이 모두 상위권을 유지하고 있어,  
   단순 물량 확대가 아니라 **효율·품질·안전이 균형 잡힌 시공 실적**을 보여줌.
"""
                )

                # 특별상 블록
                st.markdown("---")
                if special_company is None:
                    st.info("영업용 사용량 기준으로 본상과 다른 업체를 찾지 못해 특별상 후보가 없습니다.")
                else:
                    st.markdown(f"#### ⭐ 특별상(영업용 수요 확대) 추천 : **{special_company}**")

                    st.markdown(
                        f"""
- **총점 순위** : {int(special_row['총점순위'])} 위  
- **총점** : {special_row[total_col]} 점  
- **2-3항목(기존주택 비율) 점수** : {special_row[exist_col]} 점  
- **신규계량기 수** : {int(special_row["신규계량기수"]):,} 전  
- **추정 연간사용량 합계** : {fmt_int(special_row["연간사용량합계"])} m³  
- **영업용 사용량** : {fmt_int(special_row["영업용_사용량"])} m³  
- **영업용 사용량 비중** : {special_row["영업용_비중(%)"]:.1f}%  
"""
                    )

                    st.markdown(
                        """
**선정 논리 (보고서용 문장 예시)**  

1. 종합점수는 본상 업체 대비 다소 낮지만,  
   가정용 외 **영업용(식당·프랜차이즈 등) 신규 수요 개발 실적이 가장 크고 비중도 높음.**
2. 특히 자사 공급망 확대에 기여도가 큰 업종(일반음식점, 프랜차이즈 등)을 중심으로  
   신규 계량기 설치와 사용량 기반 성장이 확인되어,  
   **영업용 수요처 발굴·개척 측면에서 탁월한 공헌**을 한 것으로 평가됨.
3. 기존주택 개발 비율(2-3항목)에서도 일정 수준 이상의 점수를 유지하여,  
   신규 시장 개척과 더불어 **기존 관로 활용 측면에서도 균형 잡힌 실적**을 보임.
"""
                    )

                # 본상/특별상 용도별 세부 실적
                st.markdown("---")
                st.markdown("### 3) 본상·특별상 업체 용도별 세부 실적")

                def show_detail_block(title, company_name):
                    st.markdown(f"#### {title} – {company_name}")
                    sub = usage_by_type[usage_by_type["시공업체"] == company_name].copy()
                    if sub.empty:
                        st.info("세부 사용량 데이터가 없습니다.")
                        return
                    sub = sub.sort_values("연간사용량_추정", ascending=False)
                    sub["연간총"] = sub["연간사용량_추정"]
                    sub["추정 연간사용량(m³)"] = sub["연간총"].map(fmt_int)
                    sub["전수(전)"] = sub["전수"].map(lambda x: f"{int(x):,}")
                    disp = sub[["용도", "추정 연간사용량(m³)", "전수(전)"]]
                    st.dataframe(
                        center_style(disp),
                        use_container_width=True,
                        hide_index=True,
                    )

                show_detail_block("본상 업체 용도별 실적", main_company)
                if special_company is not None:
                    show_detail_block("특별상 업체 용도별 실적", special_company)
