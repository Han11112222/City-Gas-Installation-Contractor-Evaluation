from pathlib import Path
from io import BytesIO

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
                if pd.isna(v) or v == 0:
                    if not pd.isna(base):
                        usage[col] = base
            return float(usage.sum())

        # ── 가정용 외: 단독주택 제외 나머지 ───────
        else:
            vals = usage.dropna()
            if len(vals) == 0:
                return 0.0
            monthly_avg = float(vals.mean())
            return monthly_avg * 12.0

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
# 평가점수표(1~3-2항 + 2-3항 + 총점) 로더
#   - 같은 엑셀에 들어있는 시공업체 평가점수 시트 사용
#   - 컬럼: 최소 '구분', '총점', '2-3' 이 있어야 함
# --------------------------------------------------
def load_eval_score_table(
    uploaded_file, base_path: Path
) -> pd.DataFrame | None:
    """
    - uploaded_file 이 있으면 그 파일에서 시트를 탐색
    - 없으면 기본 DATA_FILE 에서 시트를 탐색
    - 시트 중 '구분'·'총점' 을 가지고 있고, '2-3' 으로 시작하는 컬럼이 있는 시트를 선택
    """
    try:
        if uploaded_file is not None:
            buf = BytesIO(uploaded_file.getvalue())
            xls = pd.ExcelFile(buf)
        else:
            xls = pd.ExcelFile(base_path)
    except Exception:
        return None

    target_basic = {"구분", "총점"}
    chosen = None

    for sheet in xls.sheet_names:
        df_tmp = xls.parse(sheet)
        cols = set(map(str, df_tmp.columns))
        if not target_basic.issubset(cols):
            continue

        col_23 = None
        for c in df_tmp.columns:
            name = str(c).replace(" ", "")
            if name.startswith("2-3"):
                col_23 = c
                break
        if col_23 is None:
            continue

        df_tmp = df_tmp.rename(columns={col_23: "2-3"})
        chosen = df_tmp
        break

    if chosen is None:
        return None

    # 필요 컬럼만 정리 (없으면 있는 것만 사용)
    keep_cols = [c for c in ["순번", "구분", "1-1", "2-1", "2-2", "2-3", "3-1", "3-2", "감점", "총점", "비고"] if c in chosen.columns]
    df_eval = chosen[keep_cols].copy()

    # 숫자형으로 정리
    for c in ["1-1", "2-1", "2-2", "2-3", "3-1", "3-2", "감점", "총점"]:
        if c in df_eval.columns:
            df_eval[c] = pd.to_numeric(df_eval[c], errors="coerce")

    df_eval = df_eval[df_eval["구분"].notna()].copy()
    if "총점" in df_eval.columns:
        df_eval = df_eval[df_eval["총점"].notna()]

    return df_eval


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
# 탭 4 : 최종분석 – 종합점수 + 2-3항목(기존주택 비율) 기반 포상 추천
# --------------------------------------------------
with tab_final:
    st.subheader("※ 최종분석 – 종합점수 + 2-3항목(기존주택 비율) 기반 포상 추천")

    st.markdown(
        """
- 별도의 **시공업체 평가점수표(1-1~3-2, 감점, 총점 포함)** 를 엑셀에 넣어두면  
  **총점 + 2-3항목(기존주택 개발 비율)** 점수를 활용해 **본상/특별상 후보**를 자동 추천한다.
- 기존주택 비율은 이미 평가기준 2-3항목 점수로 반영되어 있으므로,  
  이 탭에서는 별도 비율 산정 없이 **2-3 점수 컬럼**을 그대로 사용한다.
"""
    )

    # 같은 엑셀 파일 안에서 평가점수표 시트 찾기
    eval_df = load_eval_score_table(uploaded, DATA_FILE)

    if eval_df is None or eval_df.empty:
        st.info(
            "엑셀 파일에서 '구분', '2-3', '총점' 컬럼을 가진 평가점수표 시트를 찾지 못했어. "
            "점수표 시트가 같은 파일 안에 있는지 확인해줘."
        )
    else:
        # 전체 순위 (총점 기준)
        eval_rank = eval_df.copy()
        eval_rank = eval_rank.sort_values("총점", ascending=False)
        eval_rank["전체순위"] = np.arange(1, len(eval_rank) + 1)

        # 본상: 총점 1위
        main_award_row = eval_rank.iloc[0]
        main_company = str(main_award_row["구분"])
        main_total = float(main_award_row["총점"])
        main_23 = float(main_award_row["2-3"]) if "2-3" in eval_rank.columns else np.nan

        # 특별상: 2-3점수(기존주택 비율) + 외식업(식당/프랜차이즈 등) 공급 실적
        spec_df = eval_rank[["구분", "총점", "2-3"]].rename(columns={"구분": "시공업체"}).copy()

        # 외식업 관련 용도: '식당', '음식점', '프랜차이즈' 포함
        rest_rows = usage_by_type_nonres[
            usage_by_type_nonres["용도"].str.contains("식당|음식점|프랜차이즈", na=False)
        ].copy()
        if not rest_rows.empty:
            rest_agg = (
                rest_rows.groupby("시공업체")["연간사용량_추정"]
                .sum()
                .reset_index(name="외식업_연간사용량")
            )
            spec_df = spec_df.merge(rest_agg, on="시공업체", how="left")
        else:
            spec_df["외식업_연간사용량"] = np.nan

        spec_df["외식업_연간사용량"] = spec_df["외식업_연간사용량"].fillna(0.0)

        # 지수화 (0~1)
        if spec_df["2-3"].max() > 0:
            spec_df["기존주택지수"] = spec_df["2-3"] / spec_df["2-3"].max()
        else:
            spec_df["기존주택지수"] = 0.0

        if spec_df["외식업_연간사용량"].max() > 0:
            spec_df["외식업지수"] = (
                spec_df["외식업_연간사용량"]
                / spec_df["외식업_연간사용량"].max()
            )
        else:
            spec_df["외식업지수"] = 0.0

        # 특별상 종합지수: 기존주택 60%, 외식업 40%
        spec_df["특별상지수"] = (
            spec_df["기존주택지수"] * 0.6 + spec_df["외식업지수"] * 0.4
        )

        # 본상 업체는 특별상 후보에서 제외
        spec_candidates = spec_df[spec_df["시공업체"] != main_company].copy()
        special_row = spec_candidates.sort_values(
            ["특별상지수", "2-3", "총점"], ascending=False
        ).iloc[0]

        special_company = str(special_row["시공업체"])
        special_23 = float(special_row["2-3"])
        special_rest = float(special_row["외식업_연간사용량"])
        special_total = float(special_row["총점"])

        # 전체 순위표 표시 (본상/특별상 색깔 표시)
        disp_cols = []
        for c in ["전체순위", "구분", "1-1", "2-1", "2-2", "2-3", "3-1", "3-2", "감점", "총점", "비고"]:
            if c in eval_rank.columns:
                disp_cols.append(c)
        eval_display = eval_rank[disp_cols].copy()

        def highlight_awards(row):
            name = str(row["구분"])
            if name == main_company:
                color = "#FFF4CC"   # 본상: 연노랑
            elif name == special_company:
                color = "#D5F5E3"   # 특별상: 연초록
            else:
                color = ""
            return [f"background-color: {color}" if color else "" for _ in row]

        styled_eval = center_style(eval_display, highlight_awards)

        st.markdown("#### 1) 평가기준 점수표 기반 전체 순위 (총점 기준)")
        st.dataframe(
            styled_eval,
            use_container_width=True,
            hide_index=True,
            column_config={
                "전체순위": st.column_config.Column("전체순위", width="small"),
            },
        )

        # 포상 추천 요약
        st.markdown("---")
        st.markdown("#### 2) 포상 추천 결과")

        st.markdown(
            f"""
- **본상(우수 시공업체)** : `{main_company}`
  - 종합점수(총점) **{main_total:.0f}점**으로 전체 1위
  - 2-3항목(기존주택 개발 비율) 점수 **{main_23:.0f}점**으로  
    기존 주택 밀집 지역에서의 수요개발 실적도 우수
- **특별상(기존주택 + 외식업 공급 기여)** : `{special_company}`
  - 2-3항목 점수 **{special_23:.0f}점**으로 기존주택 공급 비율이 상위권
  - 외식업(식당·프랜차이즈 등) 추정 연간사용량 **{fmt_int(special_rest)} m³** 로  
    비(非)가정용 중에서도 식당·프랜차이즈 영역에서의 신규 수요 창출 기여도가 높음
  - 종합점수(총점)도 **{special_total:.0f}점**으로 상위권을 유지하여  
    **기존주택 + 상점·외식업 수요개발을 동시에 달성한 사례**로 평가 가능
"""
        )

        st.markdown("---")
        st.markdown("#### 3) 항목별 시사점 정리")

        st.markdown(
            """
- **2-3항목(기존주택 개발 비율)**  
  - 기존 배관이 깔려 있는 지역에서 추가로 신규 계량기를 설치한 비율을 의미하며,  
    동일 물량 대비 투자비와 공사 난이도가 상대적으로 낮아 **회사 입장에서는 공급 효율이 높은 실적**으로 해석할 수 있다.
- **외식업(식당·프랜차이즈 등) 공급 실적**  
  - 단위 고객당 사용량이 가정용보다 크고, 장기 계약이 유지되는 경향이 있어  
    **안정적인 판매량·매출 기반을 만들어 주는 수요**로 볼 수 있다.
- 따라서,  
  - 총점 1위 업체는 **전체적인 종합 역량(수요개발·품질·관리)** 이 가장 우수한 사례로 **본상**을 부여하고,  
  - 2-3항목과 외식업 사용량 지수를 함께 고려했을 때 두 지표 모두 의미 있게 높은 업체를 **특별상**으로 선정하는 방식이  
    회사의 **공급 효율성 제고 + 전략용도(외식업) 집중 육성**이라는 두 가지 목표를 동시에 충족시키는 포상 체계가 된다.
"""
        )


# --------------------------------------------------
# 메인 함수 (Streamlit 실행 시 필요하면 사용)
# --------------------------------------------------
# Streamlit Cloud / `streamlit run app.py` 환경에선
# 상단 코드가 곧바로 실행되므로 별도 main() 은 생략해도 된다.
