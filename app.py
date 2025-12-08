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


def center_styler(styler: pd.io.formats.style.Styler):
    """표 전체 가운데 정렬"""
    styler = styler.set_table_styles(
        [dict(selector="th", props=[("text-align", "center")])]
    ).set_properties(**{"text-align": "center"})
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

    # 먼저 가정용/가정용외 구분된 상태에서 계량기별 연간 사용량 계산
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
st.title("📊 도시가스 신규계량기 사용량 기반 우수 시공업체 평가")

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
uploaded = st.file_uploader("📁 엑셀 파일 업로드 (없으면 기본 파일 사용)", type=["xlsx"])
if uploaded is not None:
    raw_df = pd.read_excel(uploaded)
else:
    raw_df = load_raw(DATA_FILE)

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
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("전체 시공업체 수 (1종)", f"{agg_all.shape[0]:,} 개")
with col2:
    st.metric("포상 기준 충족 업체 수", f"{eligible.shape[0]:,} 개")
with col3:
    st.metric(
        "전체 신규계량기 수 (아파트 제외)",
        f"{df_proc['계량기번호'].nunique():,} 전",
    )

tab_rank, tab_type, tab_detail, tab_raw = st.tabs(
    ["업체별 순위", "용도별 분석", "업체별 용도 분석", "원자료(가공 후)"]
)

# --------------------------------------------------
# 탭 1 : 업체별 순위
# --------------------------------------------------
with tab_rank:
    st.subheader("📈 포상 대상 업체 순위 (연간 사용량 기준)")

    if eligible.empty:
        st.info("포상 기준(10전 이상 & 연간 100,000 m³ 이상)을 만족하는 업체가 없습니다.")
    else:
        rank_df = (
            eligible.reset_index()
            .loc[
                :,
                [
                    "순위",
                    "시공업체",
                    "신규계량기수",
                    "연간사용량합계",
                    "계량기당_평균연간사용량",
                ],
            ]
            .copy()
        )
        rank_df = rank_df.rename(columns={"시공업체": "시공업체명"})

        # 표시용 DataFrame
        disp = rank_df[["순위", "시공업체명", "신규계량기수",
                        "연간사용량합계", "계량기당_평균연간사용량"]].copy()
        disp = disp.rename(
            columns={
                "신규계량기수": "신규계량기 수(전)",
                "연간사용량합계": "추정 연간사용량 합계(m³)",
                "계량기당_평균연간사용량": "계량기당 평균 연간사용량(m³)",
            }
        )

        styler = disp.style.format(
            {
                "신규계량기 수(전)": "{:,.0f}",
                "추정 연간사용량 합계(m³)": "{:,.0f}",
                "계량기당 평균 연간사용량(m³)": "{:,.0f}",
            }
        )
        styler = center_styler(styler)

        st.dataframe(
            styler,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "※ 포상 기본 전제 : 연간 신규계량기 수 10전 이상, 추정 연간사용량 합계 100,000 m³ 이상일 때만 순위에 포함"
        )

        # 상위 20개 업체 바 차트
        top_n = min(20, rank_df.shape[0])
        chart_df = rank_df.head(top_n)
        fig = px.bar(
            chart_df,
            x="시공업체명",
            y="연간사용량합계",
            text=chart_df["연간사용량합계"].map(fmt_int),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="시공업체",
            yaxis_title="추정 연간사용량 합계(m³)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 전체 업체 순위 (포상 기준 미적용)
        with st.expander("📊 포상 기준 미적용 전체 업체 순위 보기"):
            all_rank = agg_all.sort_values(
                "연간사용량합계", ascending=False
            ).reset_index()
            all_rank["순위"] = np.arange(1, len(all_rank) + 1)

            disp_all = all_rank[
                ["순위", "시공업체", "신규계량기수", "연간사용량합계",
                 "계량기당_평균연간사용량"]
            ].copy()
            disp_all = disp_all.rename(
                columns={
                    "신규계량기수": "신규계량기 수(전)",
                    "연간사용량합계": "추정 연간사용량 합계(m³)",
                    "계량기당_평균연간사용량": "계량기당 평균 연간사용량(m³)",
                }
            )

            def highlight_eligible(row):
                cond = (row["신규계량기 수(전)"] >= MIN_METERS) and (
                    row["추정 연간사용량 합계(m³)"] >= MIN_ANNUAL
                )
                return [
                    "background-color: #FFF7CC" if cond else ""
                    for _ in row
                ]

            styler_all = disp_all.style.format(
                {
                    "신규계량기 수(전)": "{:,.0f}",
                    "추정 연간사용량 합계(m³)": "{:,.0f}",
                    "계량기당 평균 연간사용량(m³)": "{:,.0f}",
                }
            )
            styler_all = center_styler(styler_all)
            styler_all = styler_all.apply(highlight_eligible, axis=1)

            st.dataframe(
                styler_all,
                use_container_width=True,
                hide_index=True,
            )

        # 추가 분석: 상위 10개 업체 집중도
        st.markdown("---")
        st.markdown("📌 상위 업체 집중도 분석")
        st.markdown(
            f"- 전체 1종 시공업체의 추정 연간사용량 합계는 **{fmt_int(total_usage_all)} m³** 입니다.\n"
            f"- 이 중 상위 10개 업체가 차지하는 비중은 약 **{top10_share * 100:,.1f}%** 입니다."
        )

# --------------------------------------------------
# 탭 2 : 용도별 분석 (가정용 vs 가정용외)
# --------------------------------------------------
with tab_type:
    st.subheader("📊 대분류별 사용량 요약 (가정용 vs 가정용외)")

    # 가정용(단독주택) / 가정용외(단독·공동 제외 나머지)
    df_home = df_proc[df_proc["용도"] == "단독주택"].copy()
    df_nonres_rows = df_proc[
        (df_proc["용도"] != "단독주택") & (df_proc["용도"] != "공동주택")
    ].copy()

    rows = [
        {
            "대분류": "가정용(단독주택)",
            "계량기 수(전)": df_home["계량기번호"].nunique(),
            "추정 연간사용량(m³)": df_home["연간사용량_추정"].sum(),
        },
        {
            "대분류": "가정용외(단독·공동 제외)",
            "계량기 수(전)": df_nonres_rows["계량기번호"].nunique(),
            "추정 연간사용량(m³)": df_nonres_rows["연간사용량_추정"].sum(),
        },
        {
            "대분류": "합계",
            "계량기 수(전)": df_proc["계량기번호"].nunique(),
            "추정 연간사용량(m³)": df_proc["연간사용량_추정"].sum(),
        },
    ]
    big_df = pd.DataFrame(rows)

    styler_big = big_df.style.format(
        {
            "계량기 수(전)": "{:,.0f}",
            "추정 연간사용량(m³)": "{:,.0f}",
        }
    )
    styler_big = center_styler(styler_big)

    st.dataframe(
        styler_big,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("📊 대분류별·용도별 시공업체 순위")

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

            disp_home = res[
                ["순위", "시공업체", "연간사용량_추정", "전수"]
            ].copy()
            disp_home = disp_home.rename(
                columns={
                    "연간사용량_추정": "추정 연간사용량(m³)",
                    "전수": "전수(전)",
                }
            )

            styler_home = disp_home.style.format(
                {
                    "추정 연간사용량(m³)": "{:,.0f}",
                    "전수(전)": "{:,.0f}",
                }
            )
            styler_home = center_styler(styler_home)

            st.dataframe(
                styler_home,
                use_container_width=True,
                hide_index=True,
            )

            top_n = min(15, res.shape[0])
            chart_res = res.head(top_n)
            fig_res = px.bar(
                chart_res,
                x="시공업체",
                y="연간사용량_추정",
                text=chart_res["연간사용량_추정"].map(fmt_int),
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

            disp_nonres_comp = nonres_comp[
                ["순위", "시공업체", "연간사용량_추정", "전수"]
            ].copy()
            disp_nonres_comp = disp_nonres_comp.rename(
                columns={
                    "연간사용량_추정": "추정 연간사용량(m³)",
                    "전수": "전수(전)",
                }
            )

            styler_nonres_comp = disp_nonres_comp.style.format(
                {
                    "추정 연간사용량(m³)": "{:,.0f}",
                    "전수(전)": "{:,.0f}",
                }
            )
            styler_nonres_comp = center_styler(styler_nonres_comp)

            st.dataframe(
                styler_nonres_comp,
                use_container_width=True,
                hide_index=True,
            )

            top_n2 = min(15, nonres_comp.shape[0])
            chart_nonres = nonres_comp.head(top_n2)
            fig_nonres = px.bar(
                chart_nonres,
                x="시공업체",
                y="연간사용량_추정",
                text=chart_nonres["연간사용량_추정"].map(fmt_int),
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
        st.markdown("📌 가정용외 용도별 1위 시공업체 요약")

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
            type_disp = type_disp.rename(
                columns={
                    "시공업체": "1위 시공업체",
                    "연간사용량_추정": "1위 연간사용량(m³)",
                    "전수": "1위 전수(전)",
                }
            )
            styler_type = type_disp.style.format(
                {
                    "총연간사용량": "{:,.0f}",
                    "1위 연간사용량(m³)": "{:,.0f}",
                    "1위 전수(전)": "{:,.0f}",
                }
            )
            styler_type = center_styler(styler_type)

            st.dataframe(
                styler_type,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.markdown("📊 가정용외 용도별 시공업체 순위")

            type_list_nonres = sorted(type_disp["용도"].unique().tolist())
            selected_type = st.selectbox(
                "용도 선택 (가정용외)", type_list_nonres
            )

            sub = usage_by_type_nonres[
                usage_by_type_nonres["용도"] == selected_type
            ].copy()
            sub = sub.sort_values("연간사용량_추정", ascending=False)
            sub["순위"] = np.arange(1, len(sub) + 1)

            disp_sub = sub[
                ["순위", "시공업체", "연간사용량_추정", "전수"]
            ].copy()
            disp_sub = disp_sub.rename(
                columns={
                    "연간사용량_추정": "추정 연간사용량(m³)",
                    "전수": "전수(전)",
                }
            )

            # 포상 기준 충족 업체 하이라이트 (전수·연간사용량 기준)
            def highlight_eligible_type(row):
                cond = (row["전수(전)"] >= MIN_METERS) and (
                    row["추정 연간사용량(m³)"] >= MIN_ANNUAL
                )
                return [
                    "background-color: #FFF7CC" if cond else ""
                    for _ in row
                ]

            styler_sub = disp_sub.style.format(
                {
                    "추정 연간사용량(m³)": "{:,.0f}",
                    "전수(전)": "{:,.0f}",
                }
            )
            styler_sub = center_styler(styler_sub)
            styler_sub = styler_sub.apply(highlight_eligible_type, axis=1)

            st.dataframe(
                styler_sub,
                use_container_width=True,
                hide_index=True,
            )

            top_n_type = min(15, sub.shape[0])
            chart_type = sub.head(top_n_type)
            fig_type = px.bar(
                chart_type,
                x="시공업체",
                y="연간사용량_추정",
                text=chart_type["연간사용량_추정"].map(fmt_int),
            )
            fig_type.update_traces(textposition="outside")
            fig_type.update_layout(
                xaxis_title="시공업체",
                yaxis_title=f"{selected_type} 추정 연간사용량(m³)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_type, use_container_width=True)

            # 선택 용도 상세 리스트 (모든 용도 공통)
            st.markdown("---")
            st.markdown("📄 선택 용도별 상세 리스트 (시공업체별 시공 내역)")

            company_list = sub["시공업체"].tolist()
            selected_company_type = st.selectbox(
                f"{selected_type} 시공업체 선택", company_list
            )

            detail = df_proc[
                (df_proc["용도"] == selected_type)
                & (df_proc["시공업체"] == selected_company_type)
            ].copy()

            # 전체 리스트 (연간사용량 0도 포함), 연간사용량 내림차순
            if not detail.empty:
                detail = detail.sort_values("연간사용량_추정", ascending=False)

            if detail.empty:
                st.info("선택한 시공업체의 해당 용도 시공 내역이 없습니다.")
            else:
                detail_disp = detail[
                    ["계량기번호", "고객명", "주소", "자체업종명", "연간사용량_추정"]
                ].copy()
                detail_disp = detail_disp.rename(
                    columns={"연간사용량_추정": "연간사용량_추정(m³)"}
                )
                styler_detail = detail_disp.style.format(
                    {"연간사용량_추정(m³)": "{:,.0f}"}
                )
                styler_detail = center_styler(styler_detail)

                st.dataframe(
                    styler_detail,
                    use_container_width=True,
                    hide_index=True,
                )

# --------------------------------------------------
# 탭 3 : 업체별 용도 분석
# --------------------------------------------------
with tab_detail:
    st.subheader("📊 업체별 용도별 사용 패턴")

    if eligible.empty:
        st.info("포상 기준을 만족하는 업체가 없어서 상세 분석 대상이 없습니다.")
    else:
        target_companies = eligible.index.tolist()
        selected_company = st.selectbox(
            "시공업체 선택 (포상 대상 업체 기준)",
            target_companies,
            index=0,
        )

        comp_df = usage_by_type[
            usage_by_type["시공업체"] == selected_company
        ].copy()
        comp_df = comp_df.sort_values("연간사용량_추정", ascending=False)

        disp_comp = comp_df[
            ["용도", "연간사용량_추정", "전수"]
        ].copy()
        disp_comp = disp_comp.rename(
            columns={
                "연간사용량_추정": "추정 연간사용량(m³)",
                "전수": "전수(전)",
            }
        )
        styler_comp = disp_comp.style.format(
            {
                "추정 연간사용량(m³)": "{:,.0f}",
                "전수(전)": "{:,.0f}",
            }
        )
        styler_comp = center_styler(styler_comp)

        st.markdown(f"**선택한 시공업체 : {selected_company}**")

        fig2 = px.bar(
            comp_df,
            x="용도",
            y="연간사용량_추정",
            text=comp_df["연간사용량_추정"].map(fmt_int),
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            xaxis_title="용도",
            yaxis_title="추정 연간사용량(m³)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            styler_comp,
            use_container_width=True,
            hide_index=True,
        )

# --------------------------------------------------
# 탭 4 : 가공 후 원자료
# --------------------------------------------------
with tab_raw:
    st.subheader("📂 계량기별 가공 데이터(연간 사용량 포함)")

    show_cols = [
        "계량기번호",
        "시공업체",
        "고객명",
        "자체업종명",
        "용도",
        "대분류",
        "연간사용량_추정",
    ] + month_cols
    show_cols = [c for c in show_cols if c in df_proc.columns]

    df_show = df_proc[show_cols].copy()
    df_show = df_show.rename(
        columns={"연간사용량_추정": "연간사용량_추정(m³)"}
    )

    styler_show = df_show.style.format(
        {"연간사용량_추정(m³)": "{:,.0f}"}
    )
    styler_show = center_styler(styler_show)

    st.dataframe(
        styler_show,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "- 분석 대상은 가스시공업 **제1종** 시공업체.\n"
        "- 아파트(자체업종명)는 계산에서 제외.\n"
        "- 연립주택·다세대주택은 용도를 단독주택으로 변경하여 계산.\n"
        "- **가정용(단독주택)** 은 월 사용량의 블랭크·0을 단독주택 월평균표로 채운 뒤 1~12월 합계로 연간 사용량 산정.\n"
        "- **가정용 외**는 단독주택을 제외한 나머지 용도에 대해, 값이 있는 달의 평균(합계 / 값이 있는 달 수)에 12개월을 곱해 연간 사용량을 추정.\n"
        "- 포상 기준은 연간 신규계량기 수 10전 이상, 추정 연간사용량 100,000 m³ 이상."
    )
