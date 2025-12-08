from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="우수 시공업체 평가 대시보드",
    layout="wide",
)

# 깃허브에 올릴 때, 엑셀 파일명을 이 이름으로 맞춰서 두면 됨
DATA_FILE = Path(__file__).parent / "20251204-수요개발_신규계량기사용량현황.xlsx"

# 단독주택 월별 평균사용량 (2024년 기준, 부피)
# 1~12월: 96, 92, 67, 41, 25, 16, 9, 8, 7, 9, 21, 55
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
MIN_ANNUAL = 10_000   # 연간 10,000 m³ 이상


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_raw(path: Path) -> pd.DataFrame:
    """엑셀 원자료 로딩"""
    return pd.read_excel(path)


def get_month_cols(df: pd.DataFrame):
    """연-월(예: 202501, 202412) 형식의 월별 사용량 컬럼만 추출"""
    return [c for c in df.columns if isinstance(c, (int, np.integer))]


def build_detached_avg_by_col(month_cols):
    """단독주택 월평균을 실제 컬럼명(202501 등)에 매핑"""
    mapping = {}
    for col in month_cols:
        month_num = int(str(col)[-2:])  # 202501 -> 1, 202412 -> 12
        mapping[col] = SINGLE_DETACHED_MONTHLY_AVG.get(month_num, np.nan)
    return mapping


def preprocess(df_raw: pd.DataFrame):
    """사용 예정량 계산 + 업체별 집계 + 용도별 집계까지 한 번에 처리"""
    df = df_raw.copy()

    month_cols = get_month_cols(df)
    detached_avg_by_col = build_detached_avg_by_col(month_cols)

    # 1) 아파트는 분석 대상에서 제외
    df = df[df["자체업종명"] != "아파트"].copy()

    # 2) 연립주택, 다세대주택은 용도를 단독주택으로 변경
    mask_multi = df["자체업종명"].isin(["연립주택", "다세대주택"])
    df.loc[mask_multi, "용도"] = "단독주택"

    # 사용여부가 Y 인 계량기만 사용 (혹시 모를 예외 대비)
    if "사용여부" in df.columns:
        df = df[df["사용여부"] == "Y"].copy()

    # 3) 계량기별 연간사용량 계산 ---------------------------------
    def compute_annual(row):
        usage = row[month_cols].astype(float)

        # 단독주택: 결측/0 은 단독주택 월평균으로 대체 후 단순 합산
        if row["용도"] == "단독주택":
            for col in month_cols:
                base = detached_avg_by_col.get(col)
                v = usage[col]
                if pd.isna(v) or v == 0:
                    if not pd.isna(base):  # 월평균 값이 있을 때만 치환
                        usage[col] = base
            return float(usage.sum())

        # 공동주택: 실제 사용량 기준 합산 (결측만 0으로)
        elif row["용도"] == "공동주택":
            return float(usage.fillna(0).sum())

        # 그 외 용도(업무용, 영업용, 산업용, 열병합용 등):
        #   사용이 있는 달들의 평균 사용량 × 12개월
        else:
            vals = usage.replace(0, np.nan).dropna()
            if len(vals) == 0:
                return 0.0
            monthly_avg = float(vals.mean())
            return monthly_avg * 12.0

    df["연간사용량_추정"] = df.apply(compute_annual, axis=1)

    # 4) 시공업체별 집계 ------------------------------------------
    agg = (
        df.groupby("시공업체", as_index=True)
        .agg(
            신규계량기수=("계량기번호", "nunique"),
            연간사용량합계=("연간사용량_추정", "sum"),
        )
    )
    agg["계량기당_평균연간사용량"] = agg["연간사용량합계"] / agg["신규계량기수"]

    # 포상 대상 필터
    eligible = agg[
        (agg["신규계량기수"] >= MIN_METERS)
        & (agg["연간사용량합계"] >= MIN_ANNUAL)
    ].copy()

    eligible = eligible.sort_values("연간사용량합계", ascending=False)
    eligible["순위"] = np.arange(1, len(eligible) + 1)

    # 용도별 사용비중 분석용 피벗
    usage_by_type = (
        df.groupby(["시공업체", "용도"])["연간사용량_추정"]
        .sum()
        .reset_index()
    )

    return df, agg, eligible, usage_by_type, month_cols


# --------------------------------------------------
# UI 구성
# --------------------------------------------------
st.title("도시가스 신규계량기 사용량 기반 우수 시공업체 평가")

st.markdown(
    """
- **대상 데이터** : 수요개발 신규계량기 사용량 현황(엑셀)
- **포상 기본 전제**
  - 연간 신규계량기 수 **10전 이상**
  - 추정 연간사용량 합계 **10,000 m³ 이상**
"""
)

# 파일 업로드 (없으면 기본 파일 사용)
uploaded = st.file_uploader("엑셀 파일 업로드 (없으면 기본 파일 사용)", type=["xlsx"])

if uploaded is not None:
    raw_df = pd.read_excel(uploaded)
else:
    raw_df = load_raw(DATA_FILE)

df_proc, agg_all, eligible, usage_by_type, month_cols = preprocess(raw_df)

# 상단 KPI 카드
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("전체 시공업체 수", f"{agg_all.shape[0]:,} 개")
with col2:
    st.metric("포상 기준 충족 업체 수", f"{eligible.shape[0]:,} 개")
with col3:
    st.metric("전체 신규계량기 수(아파트 제외)", f"{df_proc['계량기번호'].nunique():,} 전")

tab_rank, tab_detail, tab_raw = st.tabs(
    ["업체별 순위", "업체별 용도 분석", "원자료(가공 후)"]
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
            .loc[:, ["순위", "시공업체", "신규계량기수", "연간사용량합계", "계량기당_평균연간사용량"]]
            .copy()
        )

        rank_df["연간사용량합계"] = rank_df["연간사용량합계"].round(1)
        rank_df["계량기당_평균연간사용량"] = rank_df["계량기당_평균연간사용량"].round(1)
        rank_df = rank_df.rename(
            columns={
                "시공업체": "시공업체명",
                "신규계량기수": "신규계량기 수(전)",
                "연간사용량합계": "추정 연간사용량 합계(m³)",
                "계량기당_평균연간사용량": "계량기당 평균 연간사용량(m³)",
            }
        )

        st.dataframe(
            rank_df,
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
            y="추정 연간사용량 합계(m³)",
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
# 탭 2 : 업체별 용도 분석
# --------------------------------------------------
with tab_detail:
    st.subheader("업체별 용도별 사용 패턴")

    if eligible.empty:
        st.info("포상 기준을 만족하는 업체가 없어서 상세 분석 대상이 없습니다.")
    else:
        target_companies = eligible.index.tolist()
        selected_company = st.selectbox(
            "시공업체 선택",
            target_companies,
            index=0,
        )

        comp_df = usage_by_type[usage_by_type["시공업체"] == selected_company].copy()
        comp_df = comp_df.sort_values("연간사용량_추정", ascending=False)

        st.markdown(f"**선택한 시공업체 : {selected_company}**")

        fig2 = px.bar(
            comp_df,
            x="용도",
            y="연간사용량_추정",
            text="연간사용량_추정",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            xaxis_title="용도",
            yaxis_title="추정 연간사용량(m³)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            comp_df.rename(columns={"연간사용량_추정": "추정 연간사용량(m³)"}).round(1),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "※ 어떤 용도(단독주택, 공동주택, 영업용, 산업용, 업무용 등)에서 실적이 집중되는지 확인할 수 있습니다."
        )

# --------------------------------------------------
# 탭 3 : 가공 후 원자료
# --------------------------------------------------
with tab_raw:
    st.subheader("계량기별 가공 데이터(연간 사용량 포함)")

    show_cols = ["계량기번호", "시공업체", "고객명", "자체업종명", "용도", "연간사용량_추정"] + month_cols
    show_cols = [c for c in show_cols if c in df_proc.columns]

    st.dataframe(
        df_proc[show_cols].round(3),
        use_container_width=True,
    )

    st.caption(
        "- 아파트(자체업종명)는 계산에서 제외\n"
        "- 연립주택·다세대주택은 용도를 단독주택으로 변경\n"
        "- 단독주택의 공란·0값은 단독주택 월평균 사용량(2024년 기준)으로 대체하여 연간 사용량 산정\n"
        "- 그 외 용도(업무용, 영업용, 산업용, 열병합용 등)는 사용이 있는 달의 평균 사용량 × 12개월로 연간 사용량 추정"
    )
