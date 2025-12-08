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

# 포상 기준
MIN_METERS = 10        # 연간 10전 이상
MIN_ANNUAL = 10_000    # 연간 10,000 m³ 이상


# --------------------------------------------------
# 유틸 함수
# --------------------------------------------------
def fmt_int(x: float) -> str:
    """정수 + 천단위 콤마 포맷"""
    return f"{int(round(x)):,}"


def get_month_cols(df: pd.DataFrame):
    """202501, 202412 처럼 숫자형 연월 컬럼만 추출"""
    return [c for c in df.columns if isinstance(c, (int, np.integer))]


def build_detached_avg_by_col(month_cols):
    """연월 컬럼명에 단독주택 월평균 사용량 매핑"""
    mapping = {}
    for col in month_cols:
        month_num = int(str(col)[-2:])  # 202501 -> 1, 202412 -> 12
        mapping[col] = SINGLE_DETACHED_MONTHLY_AVG.get(month_num, np.nan)
    return mapping


# --------------------------------------------------
# 데이터 불러오기 & 전처리
# --------------------------------------------------
@st.cache_data
def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def preprocess(df_raw: pd.DataFrame):
    """
    사용 예정량 산정 로직
      - 가스시공업 제1종만 사용
      - 아파트 제외
      - 연립/다세대 -> 단독주택
      - 단독주택 : 결측/0 은 단독주택 월평균으로 치환 후 합산
      - 공동주택 : 실제 사용량 합산
      - 기타 용도 : 사용 있는 달 평균에 12개월을 곱해서 연간 사용량으로 환산
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

    # 사용여부 'Y' 만 사용
    if "사용여부" in df.columns:
        df = df[df["사용여부"] == "Y"].copy()

    # 계량기별 연간 사용량 추정
    def compute_annual(row):
        usage = row[month_cols].astype(float)

        # 단독주택: 결측/0 -> 월평균 치환 후 합산
        if row["용도"] == "단독주택":
            for col in month_cols:
                base = detached_avg_by_col.get(col)
                v = usage[col]
                if pd.isna(v) or v == 0:
                    if not pd.isna(base):
                        usage[col] = base
            return float(usage.sum())

        # 공동주택: 실제 사용량 합산(결측은 0)
        elif row["용도"] == "공동주택":
            return float(usage.fillna(0).sum())

        # 기타 용도: 사용 있는 달 평균 × 12개월
        else:
            vals = usage.replace(0, np.nan).dropna()
            if len(vals) == 0:
                return 0.0
            monthly_avg = float(vals.mean())
            return monthly_avg * 12.0

    df["연간사용량_추정"] = df.apply(compute_annual, axis=1)

    # 시공업체별 집계
    agg = (
        df.groupby("시공업체", as_index=True)
        .agg(
            신규계량기수=("계량기번호", "nunique"),
            연간사용량합계=("연간사용량_추정", "sum"),
        )
    )
    agg["계량기당_평균연간사용량"] = agg["연간사용량합계"] / agg["신규계량기수"]

    # 포상 기준 충족 업체
    eligible = agg[
        (agg["신규계량기수"] >= MIN_METERS)
        & (agg["연간사용량합계"] >= MIN_ANNUAL)
    ].copy()
    eligible = eligible.sort_values("연간사용량합계", ascending=False)
    eligible["순위"] = np.arange(1, len(eligible) + 1)

    # 업체 × 용도별 사용량
    usage_by_type = (
        df.groupby(["시공업체", "용도"])["연간사용량_추정"]
        .sum()
        .reset_index()
    )

    return df, agg, eligible, usage_by_type, month_cols


# --------------------------------------------------
# 메인 로직
# --------------------------------------------------
st.title("도시가스 신규계량기 사용량 기반 우수 시공업체 평가")

st.markdown(
    """
- **대상 데이터** : 수요개발 신규계량기 사용량 현황(엑셀)
- **분석 대상 시공업체** : 가스시공업 **제1종** 시공업체
- **포상 기본 전제**
  - 연간 신규계량기 수 **10전 이상**
  - 추정 연간사용량 합계 **10,000 m³ 이상**
"""
)

# 파일 업로드 (없으면 저장소 내 기본 파일 사용)
uploaded = st.file_uploader("엑셀 파일 업로드 (없으면 기본 파일 사용)", type=["xlsx"])
if uploaded is not None:
    raw_df = pd.read_excel(uploaded)
else:
    raw_df = load_raw(DATA_FILE)

df_proc, agg_all, eligible, usage_by_type, month_cols = preprocess(raw_df)

# 용도별 요약 및 1위 시공업체 계산
type_summary = (
    usage_by_type.groupby("용도")
    .agg(
        총연간사용량=("연간사용량_추정", "sum"),
        업체수=("시공업체", "nunique"),
    )
    .reset_index()
)
idx = usage_by_type.groupby("용도")["연간사용량_추정"].idxmax()
top_per_type = usage_by_type.loc[idx, ["용도", "시공업체", "연간사용량_추정"]]
type_summary = type_summary.merge(
    top_per_type, on="용도", how="left"
)  # 시공업체, 연간사용량_추정(1위)

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
    st.subheader("포상 대상 업체 순위 (연간 사용량 기준)")

    if eligible.empty:
        st.info("포상 기준(10전 이상 & 연간 10,000 m³ 이상)을 만족하는 업체가 없습니다.")
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
        rank_df = rank_df.rename(
            columns={
                "시공업체": "시공업체명",
                "신규계량기수": "신규계량기 수(전)",
            }
        )

        # 원본 값 따로 보관
        rank_df["연간총"] = rank_df["연간사용량합계"]
        rank_df["계량기당평균"] = rank_df["계량기당_평균연간사용량"]

        # 표시용 포맷 (정수 + 콤마)
        rank_df["추정 연간사용량 합계(m³)"] = rank_df["연간총"].map(fmt_int)
        rank_df["계량기당 평균 연간사용량(m³)"] = rank_df["계량기당평균"].map(fmt_int)

        display_cols = [
            "순위",
            "시공업체명",
            "신규계량기 수(전)",
            "추정 연간사용량 합계(m³)",
            "계량기당 평균 연간사용량(m³)",
        ]
        st.dataframe(
            rank_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "※ 포상 기본 전제 : 연간 신규계량기 수 10전 이상, 추정 연간사용량 합계 10,000 m³ 이상일 때만 순위에 포함"
        )

        # 상위 20개 업체 바 차트
        top_n = min(20, rank_df.shape[0])
        chart_df = rank_df.head(top_n)
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

        # 전체 업체 순위 (포상 기준 미적용)
        with st.expander("포상 기준 미적용 전체 업체 순위 보기"):
            all_rank = agg_all.sort_values(
                "연간사용량합계", ascending=False
            ).reset_index()
            all_rank["순위"] = np.arange(1, len(all_rank) + 1)
            all_rank["연간총"] = all_rank["연간사용량합계"]
            all_rank["계량기당평균"] = all_rank["계량기당_평균연간사용량"]
            all_rank["추정 연간사용량 합계(m³)"] = all_rank["연간총"].map(fmt_int)
            all_rank["계량기당 평균 연간사용량(m³)"] = all_rank[
                "계량기당평균"
            ].map(fmt_int)

            disp_cols = [
                "순위",
                "시공업체",
                "신규계량기수",
                "추정 연간사용량 합계(m³)",
                "계량기당 평균 연간사용량(m³)",
            ]
            st.dataframe(
                all_rank[disp_cols],
                use_container_width=True,
                hide_index=True,
            )

        # 추가 분석: 상위 10개 업체 집중도
        st.markdown("---")
        st.markdown("#### 추가 분석: 상위 업체 집중도")
        st.markdown(
            f"- 전체 1종 시공업체의 추정 연간사용량 합계는 **{fmt_int(total_usage_all)} m³** 입니다.\n"
            f"- 이 중 상위 10개 업체가 차지하는 비중은 약 **{top10_share * 100:,.1f}%** 입니다."
        )

# --------------------------------------------------
# 탭 2 : 용도별 분석
# --------------------------------------------------
with tab_type:
    st.subheader("용도별 사용량 및 1위 시공업체")

    type_disp = type_summary.copy()
    type_disp["총 연간사용량(m³)"] = type_disp["총연간사용량"].map(fmt_int)
    type_disp["1위 추정 연간사용량(m³)"] = type_disp["연간사용량_추정"].map(fmt_int)
    type_disp = type_disp.rename(
        columns={
            "용도": "용도",
            "업체수": "시공업체 수",
            "시공업체": "1위 시공업체",
        }
    )

    st.dataframe(
        type_disp[
            ["용도", "시공업체 수", "총 연간사용량(m³)", "1위 시공업체", "1위 추정 연간사용량(m³)"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### 용도별 시공업체 순위")

    type_list = sorted(usage_by_type["용도"].unique().tolist())
    selected_type = st.selectbox("용도 선택", type_list)

    sub = usage_by_type[usage_by_type["용도"] == selected_type].copy()
    sub = sub.sort_values("연간사용량_추정", ascending=False)
    sub["순위"] = np.arange(1, len(sub) + 1)
    sub["연간총"] = sub["연간사용량_추정"]
    sub["추정 연간사용량(m³)"] = sub["연간총"].map(fmt_int)

    # 용도별 순위 리스트
    st.dataframe(
        sub[["순위", "시공업체", "추정 연간사용량(m³)"]],
        use_container_width=True,
        hide_index=True,
    )

    # 상위 15개 업체까지 바 차트
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

    # 영업용 상세 리스트 (시공업체별 시공 내용)
    if selected_type == "영업용":
        st.markdown("---")
        st.markdown("#### 영업용 상세 리스트 (시공업체별 시공 내역)")

        company_list = sub["시공업체"].tolist()
        selected_company_sales = st.selectbox(
            "영업용 시공업체 선택", company_list
        )

        detail = df_proc[
            (df_proc["용도"] == "영업용")
            & (df_proc["시공업체"] == selected_company_sales)
        ].copy()

        if detail.empty:
            st.info("선택한 시공업체의 영업용 시공 내역이 없습니다.")
        else:
            detail["연간사용량_추정(m³)"] = detail["연간사용량_추정"].map(fmt_int)
            detail_cols = [
                "계량기번호",
                "고객명",
                "주소",
                "자체업종명",
                "연간사용량_추정(m³)",
            ]

            st.dataframe(
                detail[detail_cols],
                use_container_width=True,
                hide_index=True,
            )

# --------------------------------------------------
# 탭 3 : 업체별 용도 분석
# --------------------------------------------------
with tab_detail:
    st.subheader("업체별 용도별 사용 패턴")

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
        comp_df["연간총"] = comp_df["연간사용량_추정"]
        comp_df["추정 연간사용량(m³)"] = comp_df["연간총"].map(fmt_int)

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

        st.dataframe(
            comp_df[["용도", "추정 연간사용량(m³)"]],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "※ 선택한 시공업체가 어떤 용도(단독주택, 영업용, 산업용, 업무용, 열병합용 등)에 강점을 가지는지 한눈에 확인할 수 있습니다."
        )

# --------------------------------------------------
# 탭 4 : 가공 후 원자료
# --------------------------------------------------
with tab_raw:
    st.subheader("계량기별 가공 데이터(연간 사용량 포함)")

    show_cols = [
        "계량기번호",
        "시공업체",
        "고객명",
        "자체업종명",
        "용도",
        "연간사용량_추정",
    ] + month_cols
    show_cols = [c for c in show_cols if c in df_proc.columns]

    df_show = df_proc[show_cols].copy()
    df_show["연간사용량_추정(m³)"] = df_show["연간사용량_추정"].map(fmt_int)

    st.dataframe(
        df_show.drop(columns=["연간사용량_추정"]),
        use_container_width=True,
    )

    st.caption(
        "- 분석 대상은 가스시공업 **제1종** 시공업체입니다.\n"
        "- 아파트(자체업종명)는 계산에서 제외되었습니다.\n"
        "- 연립주택·다세대주택은 용도를 단독주택으로 변경하여 계산했습니다.\n"
        "- 단독주택의 공란·0값은 단독주택 월평균 사용량(2024년 기준)으로 대체하여 연간 사용량을 산정했습니다.\n"
        "- 그 외 용도(업무용, 영업용, 산업용, 열병합용 등)는 사용이 있는 달의 평균 사용량에 12개월을 곱해 연간 사용량을 추정했습니다."
    )
