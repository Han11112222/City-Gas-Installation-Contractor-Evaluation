# app.py ─ City Gas Installation Contractor Evaluation
# - 탭1: 연도별 포상 기준 충족 현황 (연도 선택) ★ 신규
# - 탭2: 업체별 순위
# - 탭3: 용도별 분석
# - 탭4: 업체별 용도 분석
# - 탭5: 최종분석 (종합점수 고정표 + 포상 표시)
# - 탭6: 연간분석 (연도별 포상대상/용도패턴/업체별 추이)

from pathlib import Path
from typing import Dict, List, Tuple

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

BASE_DIR = Path(__file__).parent

# 기준년도 기본 파일 (2025년)
DATA_FILE = BASE_DIR / "20251204-수요개발_신규계량기사용량현황.xlsx"

# 연간 분석용 연도별 파일
YEARLY_FILES = {
    2023: BASE_DIR / "20231205-수요개발_신규공급계량기사용량현황.xlsx",
    2024: BASE_DIR / "20241206-수요개발_신규공급계량기사용량현황.xlsx",
    2025: BASE_DIR / "20251204-수요개발_신규계량기사용량현황.xlsx",
}

# 단독주택 월별 평균사용량 (2024년 기준, 부피 m³)
SINGLE_DETACHED_MONTHLY_AVG = {
    1: 96, 2: 92, 3: 67, 4: 41, 5: 25, 6: 16,
    7: 9,  8: 8,  9: 7, 10: 9, 11: 21, 12: 55,
}

# 포상 기준
MIN_METERS = 10
MIN_ANNUAL = 100_000

# KPI 고정값
TOTAL_METERS_NO_APT_FIXED   = 2_891
TOTAL_METERS_INCL_APT_FIXED = 17_745
HOME_METERS_FIXED   = 2_187
NONRES_METERS_FIXED =   704
TOTAL_COMPANY_FIXED = 70

# --------------------------------------------------
# 유틸 함수
# --------------------------------------------------
def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}"

def get_month_cols(df: pd.DataFrame) -> List:
    return [c for c in df.columns if isinstance(c, (int, np.integer))]

def build_detached_avg_by_col(month_cols: List[int]) -> Dict[int, float]:
    return {col: SINGLE_DETACHED_MONTHLY_AVG.get(int(str(col)[-2:]), np.nan) for col in month_cols}

def center_style(df: pd.DataFrame, highlight_fn=None):
    styler = df.style
    if highlight_fn is not None:
        styler = styler.apply(highlight_fn, axis=1)
    styler = styler.set_properties(**{"text-align": "center"})
    styler = styler.set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )
    return styler

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------
@st.cache_data
def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()
    if "업종" in df.columns:
        df = df[df["업종"] == "가스시공업 제1종"].copy()

    month_cols = get_month_cols(df)
    detached_avg = build_detached_avg_by_col(month_cols)

    if "자체업종명" in df.columns:
        df = df[df["자체업종명"] != "아파트"].copy()

    if "자체업종명" in df.columns and "용도" in df.columns:
        df.loc[df["자체업종명"].isin(["연립주택", "다세대주택"]), "용도"] = "단독주택"

    if "사용여부" in df.columns:
        df = df[df["사용여부"] == "Y"].copy()

    annual_col = next((c for c in ["연간 예상사용량", "연간예상사용량"] if c in df.columns), None)
    if annual_col:
        df[annual_col] = pd.to_numeric(
            df[annual_col].astype(str).str.replace(",", "", regex=False).replace({"": np.nan}),
            errors="coerce"
        )

    def compute_annual(row):
        usage = row[month_cols].astype(float) if month_cols else pd.Series(dtype=float)
        if "용도" in row and row["용도"] == "단독주택":
            for col in month_cols:
                v = usage[col]
                base = detached_avg.get(col)
                if (pd.isna(v) or v == 0) and not pd.isna(base):
                    usage[col] = base
            return float(usage.sum())
        if annual_col:
            base_annual = row.get(annual_col, np.nan)
            if not pd.isna(base_annual):
                return float(base_annual)
        vals = usage.dropna()
        return float(vals.mean()) * 12.0 if len(vals) > 0 else 0.0

    df["연간사용량_추정"] = df.apply(compute_annual, axis=1)
    df["대분류"] = np.where(df.get("용도", pd.Series("가정용외", index=df.index)) == "단독주택", "가정용(단독주택)", "가정용외")

    if "시공업체" not in df.columns: df["시공업체"] = "미상"
    if "계량기번호" not in df.columns: df["계량기번호"] = df.index.astype(str)

    agg = df.groupby("시공업체").agg(
        신규계량기수=("계량기번호", "nunique"),
        연간사용량합계=("연간사용량_추정", "sum"),
    )
    agg["계량기당_평균연간사용량"] = agg["연간사용량합계"] / agg["신규계량기수"]

    eligible = agg[(agg["신규계량기수"] >= MIN_METERS) & (agg["연간사용량합계"] >= MIN_ANNUAL)].copy()
    eligible = eligible.sort_values("연간사용량합계", ascending=False)
    eligible["순위"] = np.arange(1, len(eligible) + 1)

    usage_by_type = (
        df.groupby(["시공업체", "용도"])
        .agg(연간사용량_추정=("연간사용량_추정", "sum"), 전수=("계량기번호", "nunique"))
        .reset_index()
    )
    df_nonres = df[(df["용도"] != "단독주택") & (df["용도"] != "공동주택")].copy()
    usage_by_type_nonres = (
        df_nonres.groupby(["시공업체", "용도"])
        .agg(연간사용량_추정=("연간사용량_추정", "sum"), 전수=("계량기번호", "nunique"))
        .reset_index()
    )
    return df, agg, eligible, usage_by_type, usage_by_type_nonres, month_cols

@st.cache_data
def load_yearly_dataset():
    data_by_year, years = {}, []
    for year, path in YEARLY_FILES.items():
        if path.exists():
            df_proc, agg_all, eligible, usage_by_type, usage_by_type_nonres, _ = preprocess(pd.read_excel(path))
            data_by_year[year] = {
                "df_proc": df_proc, "agg_all": agg_all, "eligible": eligible,
                "usage_by_type": usage_by_type, "usage_by_type_nonres": usage_by_type_nonres,
            }
            years.append(year)
    return data_by_year, sorted(years)

# --------------------------------------------------
# 메인 타이틀
# --------------------------------------------------
st.title("도시가스 신규계량기 사용량 기반 우수 시공업체 평가")
st.markdown("""
- **대상 데이터** : 수요개발 신규계량기 사용량 현황(엑셀)
- **분석 대상 시공업체** : 가스시공업 **제1종** 시공업체
- **포상 기본 전제** : 연간 신규계량기 수 **10전 이상** & 추정 연간사용량 합계 **100,000 m³ 이상** (아파트 제외)
""")

uploaded = st.file_uploader("엑셀 파일 업로드 (없으면 기본 파일 사용)", type=["xlsx"])
raw_df = pd.read_excel(uploaded) if uploaded else load_raw(DATA_FILE)

df_proc, agg_all, eligible, usage_by_type, usage_by_type_nonres, month_cols = preprocess(raw_df)

total_usage_all = agg_all["연간사용량합계"].sum()
top10_share = agg_all.sort_values("연간사용량합계", ascending=False)["연간사용량합계"].head(10).sum() / total_usage_all if total_usage_all > 0 else 0

# 상단 KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 시공업체 수 (1종)", f"{TOTAL_COMPANY_FIXED:,} 개")
c2.metric("포상 기준 충족 업체 수", f"{eligible.shape[0]:,} 개")
c3.metric("전체 신규계량기 수 (공동주택 제외)", f"{TOTAL_METERS_NO_APT_FIXED:,} 전")
c4.metric("전체 신규계량기 수 (공동주택 포함)", f"{TOTAL_METERS_INCL_APT_FIXED:,} 전")

# --------------------------------------------------
# 탭 구성
# --------------------------------------------------
tab_year_select, tab_rank, tab_type, tab_detail, tab_final, tab_yearly = st.tabs(
    ["📅 연도별 포상 현황", "업체별 순위", "용도별 분석", "업체별 용도 분석", "최종분석", "연간분석"]
)

# ==================================================
# 탭1: 연도별 포상 기준 충족 현황 ★ 핵심 신규
# ==================================================
with tab_year_select:
    st.subheader("📅 연도 선택 → 포상 기준 충족 업체 현황")

    data_by_year, years = load_yearly_dataset()

    if not years:
        st.warning("연도별 파일을 찾지 못했습니다.")
    else:
        selected_year = st.selectbox(
            "분석 연도 선택", years, index=len(years)-1, format_func=lambda x: f"{x}년"
        )

        info    = data_by_year[selected_year]
        agg_y   = info["agg_all"]
        elig_y  = info["eligible"]

        # KPI
        st.markdown(f"### {selected_year}년 요약")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("전체 시공업체 수 (1종)", f"{agg_y.shape[0]:,} 개")
        k2.metric("포상 기준 충족 업체 수", f"{elig_y.shape[0]:,} 개",
                  help="신규계량기 10전 이상 & 연간사용량 100,000 m³ 이상")
        k3.metric("전체 신규계량기 수 (아파트 제외)", f"{int(agg_y['신규계량기수'].sum()):,} 전")
        k4.metric("추정 연간사용량 합계", f"{fmt_int(agg_y['연간사용량합계'].sum())} m³")

        st.markdown("---")

        # 포상 기준 충족 업체 리스트
        st.markdown(f"#### 🏆 {selected_year}년 포상 기준 충족 업체 리스트")
        st.caption(f"기준: 신규계량기 **{MIN_METERS}전 이상** & 추정 연간사용량 **{MIN_ANNUAL:,} m³ 이상** (아파트 제외)")

        if elig_y.empty:
            st.info("포상 기준을 만족하는 업체가 없습니다.")
        else:
            elig_disp = elig_y.reset_index().copy()
            elig_disp["순위"] = np.arange(1, len(elig_disp)+1)
            elig_disp["신규계량기 수(전)"]    = elig_disp["신규계량기수"].map(fmt_int)
            elig_disp["추정 연간사용량(m³)"]  = elig_disp["연간사용량합계"].map(fmt_int)
            elig_disp["계량기당 평균(m³/전)"] = elig_disp["계량기당_평균연간사용량"].map(lambda x: f"{x:,.0f}")

            def highlight_top3(row):
                if row["순위"] == 1:
                    return ["background-color:#FFD700;font-weight:bold"] * len(row)
                elif row["순위"] == 2:
                    return ["background-color:#E8E8E8;font-weight:bold"] * len(row)
                elif row["순위"] == 3:
                    return ["background-color:#CD853F;color:white;font-weight:bold"] * len(row)
                return [""] * len(row)

            disp_cols = ["순위", "시공업체", "신규계량기 수(전)", "추정 연간사용량(m³)", "계량기당 평균(m³/전)"]
            st.dataframe(
                center_style(elig_disp[disp_cols], highlight_top3),
                use_container_width=True, hide_index=True,
                column_config={"순위": st.column_config.Column("순위", width="small")},
            )
            st.caption("🥇 금색=1위 / 🥈 은색=2위 / 🥉 동색=3위")

            # 막대 차트
            st.markdown("---")
            st.markdown(f"#### 📊 {selected_year}년 포상 기준 충족 업체 사용량 비교")
            fig = px.bar(
                elig_disp,
                x="시공업체", y=elig_y["연간사용량합계"].values,
                text="추정 연간사용량(m³)",
                labels={"y": "추정 연간사용량(m³)"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_title="시공업체", yaxis_title="추정 연간사용량(m³)", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 전체 업체 목록 (접기)
        with st.expander(f"📋 {selected_year}년 전체 업체 현황 (포상 기준 미달 포함)", expanded=False):
            all_y = agg_y.sort_values("연간사용량합계", ascending=False).reset_index()
            all_y["순위"] = np.arange(1, len(all_y)+1)
            all_y["신규계량기 수(전)"]   = all_y["신규계량기수"].map(fmt_int)
            all_y["추정 연간사용량(m³)"] = all_y["연간사용량합계"].map(fmt_int)
            elig_names = set(elig_y.index.tolist())

            def hl_elig(row):
                return ["background-color:#FFF4CC"] * len(row) if row["시공업체"] in elig_names else [""] * len(row)

            st.dataframe(
                center_style(all_y[["순위","시공업체","신규계량기 수(전)","추정 연간사용량(m³)"]], hl_elig),
                use_container_width=True, hide_index=True,
            )
            st.caption("노란색 행 = 포상 기준 충족 업체")

        # 연도 비교 요약
        st.markdown("---")
        st.markdown("#### 📆 연도별 포상 기준 충족 현황 비교")
        compare_rows = []
        for y in years:
            a = data_by_year[y]["agg_all"]
            e = data_by_year[y]["eligible"]
            compare_rows.append({
                "연도": f"{y}년",
                "전체 업체 수(1종)": a.shape[0],
                "포상 충족 업체 수": e.shape[0],
                "충족 비율(%)": f"{e.shape[0]/a.shape[0]*100:.1f}%" if a.shape[0]>0 else "0.0%",
                "전체 계량기 수(전)": fmt_int(a["신규계량기수"].sum()),
                "연간사용량 합계(m³)": fmt_int(a["연간사용량합계"].sum()),
            })
        compare_df = pd.DataFrame(compare_rows)

        def hl_sel_year(row):
            return ["background-color:#E3F2FD;font-weight:bold"] * len(row) if row["연도"] == f"{selected_year}년" else [""] * len(row)

        st.dataframe(center_style(compare_df, hl_sel_year), use_container_width=True, hide_index=True)
        st.caption("파란색 행 = 현재 선택된 연도")


# ==================================================
# 탭2: 업체별 순위
# ==================================================
with tab_rank:
    st.subheader("📈 포상 기준 + 전체 업체 순위 (연간 사용량 기준)")

    all_rank = agg_all.sort_values("연간사용량합계", ascending=False).reset_index()
    all_rank["순위"] = np.arange(1, len(all_rank)+1)
    all_rank["신규계량기 수(전)"]      = all_rank["신규계량기수"]
    all_rank["추정 연간사용량 합계(m³)"] = all_rank["연간사용량합계"].map(fmt_int)

    def hl_elig_rank(row):
        return ["background-color:#FFF4CC" if row["시공업체"] in eligible.index else "" for _ in row]

    st.dataframe(
        center_style(all_rank[["순위","시공업체","신규계량기 수(전)","추정 연간사용량 합계(m³)"]], hl_elig_rank),
        use_container_width=True, hide_index=True,
        column_config={"순위": st.column_config.Column("순위", width="small")},
    )
    st.caption(
        f"- 노란색 행: 포상 기준(10전 이상 & 100,000 m³ 이상) 충족 업체\n"
        f"- 전체 추정 연간사용량: **{fmt_int(total_usage_all)} m³** | 상위 10개 업체 비중: **{top10_share*100:.1f}%**"
    )

    st.markdown("---")
    st.markdown("#### 🏆 포상 기준 충족 업체 상위 사용량")
    if eligible.empty:
        st.info("포상 기준을 만족하는 업체가 없습니다.")
    else:
        rank_df = eligible.reset_index().sort_values("연간사용량합계", ascending=False).copy()
        rank_df["추정 연간사용량 합계(m³)"] = rank_df["연간사용량합계"].map(fmt_int)
        chart_df = rank_df.head(min(20, rank_df.shape[0]))
        fig = px.bar(chart_df, x="시공업체", y="연간사용량합계", text="추정 연간사용량 합계(m³)")
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="시공업체", yaxis_title="추정 연간사용량 합계(m³)", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)


# ==================================================
# 탭3: 용도별 분석
# ==================================================
with tab_type:
    st.subheader("📊 대분류별 사용량 요약 (가정용 vs 가정용외)")

    df_home    = df_proc[df_proc["용도"] == "단독주택"]
    df_nonres_rows = df_proc[(df_proc["용도"] != "단독주택") & (df_proc["용도"] != "공동주택")]
    total_m3   = df_proc["연간사용량_추정"].sum()

    big_df = pd.DataFrame([
        {"대분류": "가정용(공동주택 제외)", "계량기 수(전)": HOME_METERS_FIXED,          "추정 연간사용량(m³)": df_home["연간사용량_추정"].sum()},
        {"대분류": "가정용외",              "계량기 수(전)": NONRES_METERS_FIXED,         "추정 연간사용량(m³)": df_nonres_rows["연간사용량_추정"].sum()},
        {"대분류": "합계",                  "계량기 수(전)": TOTAL_METERS_NO_APT_FIXED,   "추정 연간사용량(m³)": total_m3},
    ])
    big_df["사용량 비중(%)"] = big_df["추정 연간사용량(m³)"] / total_m3 * 100 if total_m3 > 0 else 0
    big_df.loc[big_df["대분류"] == "합계", "사용량 비중(%)"] = 100.0
    big_df["계량기 수(전)"]      = big_df["계량기 수(전)"].map(lambda x: f"{int(x):,}")
    big_df["추정 연간사용량(m³)"] = big_df["추정 연간사용량(m³)"].map(fmt_int)
    big_df["사용량 비중(%)"]      = big_df["사용량 비중(%)"].map(lambda x: f"{x:.1f}%")
    st.dataframe(center_style(big_df), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📌 대분류별·용도별 시공업체 순위")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["가정용(단독주택) 순위", "가정용외 순위", "가정용외 용도별 분석"])

    with sub_tab1:
        res = usage_by_type[usage_by_type["용도"] == "단독주택"].copy()
        if res.empty:
            st.info("단독주택 데이터가 없습니다.")
        else:
            res = res.sort_values("연간사용량_추정", ascending=False)
            res["순위"] = np.arange(1, len(res)+1)
            res["추정 연간사용량(m³)"] = res["연간사용량_추정"].map(fmt_int)
            res["전수(전)"] = res["전수"].map(lambda x: f"{int(x):,}")
            st.dataframe(center_style(res[["순위","시공업체","추정 연간사용량(m³)","전수(전)"]]), use_container_width=True, hide_index=True)
            fig_r = px.bar(res.head(15), x="시공업체", y="연간사용량_추정", text="추정 연간사용량(m³)")
            fig_r.update_traces(textposition="outside")
            fig_r.update_layout(xaxis_title="시공업체", yaxis_title="단독주택 추정 연간사용량(m³)", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_r, use_container_width=True)

    with sub_tab2:
        nonres_comp = (
            usage_by_type_nonres.groupby("시공업체")
            .agg(연간사용량_추정=("연간사용량_추정","sum"), 전수=("전수","sum"))
            .reset_index().sort_values("연간사용량_추정", ascending=False)
        )
        if nonres_comp.empty:
            st.info("가정용외 데이터가 없습니다.")
        else:
            nonres_comp["순위"] = np.arange(1, len(nonres_comp)+1)
            nonres_comp["추정 연간사용량(m³)"] = nonres_comp["연간사용량_추정"].map(fmt_int)
            nonres_comp["전수(전)"] = nonres_comp["전수"].map(lambda x: f"{int(x):,}")
            st.dataframe(center_style(nonres_comp[["순위","시공업체","추정 연간사용량(m³)","전수(전)"]]), use_container_width=True, hide_index=True)
            fig_nr = px.bar(nonres_comp.head(15), x="시공업체", y="연간사용량_추정", text="추정 연간사용량(m³)")
            fig_nr.update_traces(textposition="outside")
            fig_nr.update_layout(xaxis_title="시공업체", yaxis_title="가정용외 추정 연간사용량(m³)", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_nr, use_container_width=True)

    with sub_tab3:
        st.markdown("##### 📌 가정용외 용도별 1위 시공업체")
        type_sum = (
            usage_by_type_nonres.groupby("용도")
            .agg(총연간사용량=("연간사용량_추정","sum"), 업체수=("시공업체","nunique"))
            .reset_index()
        )
        idx_max = usage_by_type_nonres.groupby("용도")["연간사용량_추정"].idxmax()
        top_per = usage_by_type_nonres.loc[idx_max, ["용도","시공업체","연간사용량_추정","전수"]]
        type_sum = type_sum.merge(top_per, on="용도", how="left")
        if type_sum.empty:
            st.info("가정용외 용도 데이터가 없습니다.")
        else:
            type_sum["1위 연간사용량(m³)"] = type_sum["연간사용량_추정"].map(fmt_int)
            type_sum["1위 전수(전)"]        = type_sum["전수"].map(lambda x: f"{int(x):,}")
            type_sum = type_sum.rename(columns={"시공업체": "1위 시공업체"})
            st.dataframe(center_style(type_sum[["용도","1위 시공업체","1위 연간사용량(m³)","1위 전수(전)"]]), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("##### 📌 가정용외 용도별 시공업체 순위")
            selected_type = st.selectbox("용도 선택 (가정용외)", sorted(type_sum["용도"].unique()))
            sub = usage_by_type_nonres[usage_by_type_nonres["용도"] == selected_type].copy()
            sub = sub.sort_values("연간사용량_추정", ascending=False)
            sub["순위"] = np.arange(1, len(sub)+1)
            sub["추정 연간사용량(m³)"] = sub["연간사용량_추정"].map(fmt_int)
            sub["전수(전)"] = sub["전수"].map(lambda x: f"{int(x):,}")
            st.dataframe(center_style(sub[["순위","시공업체","추정 연간사용량(m³)","전수(전)"]]), use_container_width=True, hide_index=True)
            fig_t = px.bar(sub.head(15), x="시공업체", y="연간사용량_추정", text="추정 연간사용량(m³)")
            fig_t.update_traces(textposition="outside")
            fig_t.update_layout(xaxis_title="시공업체", yaxis_title=f"{selected_type} 추정 연간사용량(m³)", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_t, use_container_width=True)

            st.markdown("---")
            st.markdown(f"##### 🧾 {selected_type} 상세 리스트")
            selected_company_type = st.selectbox(f"{selected_type} 시공업체 선택", sub["시공업체"].tolist())
            detail = df_proc[(df_proc["용도"] == selected_type) & (df_proc["시공업체"] == selected_company_type)].copy()
            if detail.empty:
                st.info("선택한 시공업체의 해당 용도 시공 내역이 없습니다.")
            else:
                detail = detail.sort_values("연간사용량_추정", ascending=False)
                detail["연간사용량_추정(m³)"] = detail["연간사용량_추정"].map(fmt_int)
                detail_cols = ["계량기번호","고객명","주소","자체업종명","연간사용량_추정(m³)"]
                exist_cols = [c for c in detail_cols if c in detail.columns]
                st.dataframe(center_style(detail[exist_cols]), use_container_width=True, hide_index=True)


# ==================================================
# 탭4: 업체별 용도 분석
# ==================================================
with tab_detail:
    st.subheader("📌 업체별 용도별 사용 패턴")
    if eligible.empty:
        st.info("포상 기준을 만족하는 업체가 없어서 상세 분석 대상이 없습니다.")
    else:
        selected_company = st.selectbox("시공업체 선택 (포상 기준 충족 업체 기준)", eligible.index.tolist(), index=0)
        comp_df = usage_by_type[usage_by_type["시공업체"] == selected_company].copy()
        comp_df = comp_df.sort_values("연간사용량_추정", ascending=False)
        comp_df["추정 연간사용량(m³)"] = comp_df["연간사용량_추정"].map(fmt_int)
        comp_df["전수(전)"] = comp_df["전수"].map(lambda x: f"{int(x):,}")
        st.markdown(f"**선택한 시공업체 : {selected_company}**")
        fig2 = px.bar(comp_df, x="용도", y="연간사용량_추정", text="추정 연간사용량(m³)")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(xaxis_title="용도", yaxis_title="추정 연간사용량(m³)", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(center_style(comp_df[["용도","추정 연간사용량(m³)","전수(전)"]]), use_container_width=True, hide_index=True)


# ==================================================
# 탭5: 최종분석
# ==================================================
with tab_final:
    st.subheader("※ 최종분석 - 종합점수 기반 포상 추천")
    st.markdown("""
- 별도 업로드 없이 **평가점수표에서 산정한 최종 점수**를 그대로 사용합니다.
- 2-3항목(기존주택 비율)은 이미 `수요개발관리` 점수 안에 반영되어 있으므로
  이 탭에서는 별도 비율 계산 없이 **총점 기준 순위**만 사용합니다.
""")
    eval_data = [
        {"순번":1,"업체":"보민에너지(주)",       "경영일반":3,"수요개발관리":34,"품질관리":41,"감점":0,"총점":78,"순위":1},
        {"순번":2,"업체":"(주)대경지엔에스",     "경영일반":3,"수요개발관리":18,"품질관리":45,"감점":0,"총점":66,"순위":2},
        {"순번":3,"업체":"주식회사 유성산업개발","경영일반":3,"수요개발관리":26,"품질관리":37,"감점":0,"총점":66,"순위":2},
        {"순번":4,"업체":"(주)영화이엔지",       "경영일반":4,"수요개발관리":14,"품질관리":43,"감점":0,"총점":61,"순위":4},
        {"순번":5,"업체":"디에스이앤씨(주)",     "경영일반":5,"수요개발관리":34,"품질관리":16,"감점":0,"총점":55,"순위":5},
        {"순번":6,"업체":"주식회사삼주이엔지",   "경영일반":4,"수요개발관리":16,"품질관리":30,"감점":0,"총점":50,"순위":6},
        {"순번":7,"업체":"(주)신한설비",         "경영일반":4,"수요개발관리":18,"품질관리":17,"감점":0,"총점":39,"순위":7},
        {"순번":8,"업체":"동우에너지주식회사",   "경영일반":2,"수요개발관리":14,"품질관리":23,"감점":0,"총점":39,"순위":7},
        {"순번":9,"업체":"금강에너지 주식회사",  "경영일반":2,"수요개발관리":14,"품질관리":23,"감점":0,"총점":39,"순위":7},
    ]
    eval_df = pd.DataFrame(eval_data)
    eval_df["포상"] = eval_df["순위"].apply(lambda x: "포상" if x == 1 else "")

    def hl_awards(row):
        return ["background-color:#FFF4CC"] * len(row) if row["순위"] == 1 else [""] * len(row)

    st.dataframe(
        center_style(eval_df[["순번","업체","경영일반","수요개발관리","품질관리","감점","총점","순위","포상"]], hl_awards),
        use_container_width=True, hide_index=True,
        column_config={
            "순번": st.column_config.Column("순번", width="small"),
            "순위": st.column_config.Column("순위", width="small"),
        },
    )
    st.caption("- 노란색 행이 **포상 대상(1위 업체)**.")


# ==================================================
# 탭6: 연간분석
# ==================================================
with tab_yearly:
    st.subheader("📆 연간 추이 분석")

    data_by_year_t6, years_t6 = load_yearly_dataset()

    if not years_t6:
        st.info("연간 분석에 사용할 연도별 파일을 찾지 못했습니다.")
    else:
        st.markdown(f"- 분석 대상 연도: **{', '.join(map(str, years_t6))}년**")
        sub1, sub2, sub3, sub4 = st.tabs(["연도별 포상대상 현황", "연도별 용도 패턴", "업체별 연간 실적 추이", "연도별 Top-N 업체"])

        with sub1:
            st.markdown("#### 🏆 연도별 포상 기준 충족 현황")
            rows = []
            for y in years_t6:
                a = data_by_year_t6[y]["agg_all"]
                e = data_by_year_t6[y]["eligible"]
                rows.append({
                    "연도": y,
                    "전체 시공업체 수(1종)": a.shape[0],
                    "포상 기준 충족 업체 수": e.shape[0],
                    "포상 기준 충족 비율(%)": e.shape[0]/a.shape[0]*100 if a.shape[0]>0 else 0,
                    "전체 신규계량기 수(전)": a["신규계량기수"].sum(),
                    "추정 연간사용량 합계(m³)": a["연간사용량합계"].sum(),
                })
            ys = pd.DataFrame(rows)
            disp = ys.copy()
            disp["전체 신규계량기 수(전)"]      = disp["전체 신규계량기 수(전)"].map(fmt_int)
            disp["추정 연간사용량 합계(m³)"]    = disp["추정 연간사용량 합계(m³)"].map(fmt_int)
            disp["포상 기준 충족 비율(%)"]      = disp["포상 기준 충족 비율(%)"].map(lambda x: f"{x:.1f}%")
            st.dataframe(center_style(disp), use_container_width=True, hide_index=True)

            fig_l1 = px.line(ys, x="연도", y=["전체 시공업체 수(1종)","포상 기준 충족 업체 수"], markers=True)
            fig_l1.update_layout(yaxis_title="업체 수(개)", legend_title="구분", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_l1, use_container_width=True)

            fig_l2 = px.line(ys, x="연도", y="포상 기준 충족 비율(%)", markers=True)
            fig_l2.update_layout(yaxis_title="포상 기준 충족 비율(%)", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_l2, use_container_width=True)

        with sub2:
            st.markdown("#### 🔍 연도별 용도별 사용량 패턴 변화")
            rows_cat, rows_type = [], []
            for y in years_t6:
                df_y = data_by_year_t6[y]["df_proc"]
                dh   = df_y[df_y["용도"] == "단독주택"]
                dnr  = df_y[(df_y["용도"] != "단독주택") & (df_y["용도"] != "공동주택")]
                rows_cat += [
                    {"연도": y, "대분류": "가정용(공동주택 제외)", "연간사용량(m³)": dh["연간사용량_추정"].sum()},
                    {"연도": y, "대분류": "가정용외",              "연간사용량(m³)": dnr["연간사용량_추정"].sum()},
                ]
                ut = dnr.groupby("용도")["연간사용량_추정"].sum().reset_index()
                ut["연도"] = y
                rows_type.append(ut)

            cat_df = pd.DataFrame(rows_cat)
            fig_cat = px.bar(cat_df, x="연도", y="연간사용량(m³)", color="대분류", barmode="group", text="연간사용량(m³)")
            fig_cat.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_cat.update_layout(yaxis_title="연간사용량(m³)", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_cat, use_container_width=True)

            type_df = pd.concat(rows_type, ignore_index=True) if rows_type else None
            st.markdown("---")
            st.markdown("##### 📎 가정용외 세부 용도별 추세")
            if type_df is None or type_df.empty:
                st.info("가정용외 세부 용도 데이터가 없습니다.")
            else:
                top_types = type_df.groupby("용도")["연간사용량_추정"].sum().sort_values(ascending=False).head(10).index.tolist()
                sel_types = st.multiselect("비교할 용도 선택", top_types, default=top_types[:3])
                if sel_types:
                    fig_ty = px.line(type_df[type_df["용도"].isin(sel_types)], x="연도", y="연간사용량_추정", color="용도", markers=True)
                    fig_ty.update_layout(yaxis_title="연간사용량(m³)", margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_ty, use_container_width=True)

        with sub3:
            st.markdown("#### 🏗 업체별 연간 실적 추이")
            comp_set = set()
            for y in years_t6:
                comp_set.update(data_by_year_t6[y]["agg_all"].index.tolist())
            sel_comp = st.selectbox("시공업체 선택", sorted(comp_set))
            rows_comp = []
            for y in years_t6:
                a = data_by_year_t6[y]["agg_all"]
                if sel_comp in a.index:
                    r = a.loc[sel_comp]
                    rows_comp.append({"연도": y, "신규계량기 수(전)": r["신규계량기수"], "추정 연간사용량 합계(m³)": r["연간사용량합계"]})
                else:
                    rows_comp.append({"연도": y, "신규계량기 수(전)": 0, "추정 연간사용량 합계(m³)": 0})
            ct = pd.DataFrame(rows_comp)
            ca, cb = st.columns(2)
            with ca:
                fig_m = px.line(ct, x="연도", y="신규계량기 수(전)", markers=True)
                fig_m.update_layout(yaxis_title="신규계량기 수(전)", margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_m, use_container_width=True)
            with cb:
                fig_u = px.line(ct, x="연도", y="추정 연간사용량 합계(m³)", markers=True)
                fig_u.update_layout(yaxis_title="추정 연간사용량 합계(m³)", margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_u, use_container_width=True)
            st.dataframe(
                center_style(ct.assign(**{
                    "신규계량기 수(전)": ct["신규계량기 수(전)"].map(fmt_int),
                    "추정 연간사용량 합계(m³)": ct["추정 연간사용량 합계(m³)"].map(fmt_int),
                })),
                use_container_width=True, hide_index=True,
            )

        with sub4:
            st.markdown("#### 🌟 연도별 Top-N 포상 후보 비교")
            year_sel = st.selectbox("연도 선택", years_t6, index=len(years_t6)-1)
            top_n    = st.slider("Top-N 범위 선택", min_value=3, max_value=15, value=10)
            ay = data_by_year_t6[year_sel]["agg_all"].copy().sort_values("연간사용량합계", ascending=False).head(top_n).reset_index()
            ay["추정 연간사용량 합계(m³)"] = ay["연간사용량합계"].map(fmt_int)
            ay["신규계량기 수(전)"]         = ay["신규계량기수"].map(fmt_int)
            ay["순위"] = np.arange(1, len(ay)+1)
            st.dataframe(center_style(ay[["순위","시공업체","신규계량기 수(전)","추정 연간사용량 합계(m³)"]]), use_container_width=True, hide_index=True)
            fig_top = px.bar(ay, x="시공업체", y="연간사용량합계", text="추정 연간사용량 합계(m³)")
            fig_top.update_traces(textposition="outside")
            fig_top.update_layout(xaxis_title="시공업체", yaxis_title="추정 연간사용량 합계(m³)", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_top, use_container_width=True)# app.py ─ City Gas Installation Contractor Evaluation
# - 탭1: 업체별 순위
# - 탭2: 용도별 분석
# - 탭3: 업체별 용도 분석
# - 탭4: 최종분석 (종합점수 고정표 + 포상 표시)
# - 탭5: 연간분석 (연도별 포상대상/용도패턴/업체별 추이)

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="도시가스 신규계량기 사용량 기반 우수 시공업체 평가",
    layout="wide",
)

BASE_DIR = Path(__file__).parent

# 기준년도 기본 파일 (2025년)
DATA_FILE = BASE_DIR / "20251204-수요개발_신규계량기사용량현황.xlsx"

# 연간 분석용 연도별 파일 (폴더 안에 있는 실제 파일명을 맞춰서 사용)
YEARLY_FILES = {
    2023: BASE_DIR / "20231205-수요개발_신규공급계량기사용량현황.xlsx",
    2024: BASE_DIR / "20241206-수요개발_신규공급계량기사용량현황.xlsx",
    2025: BASE_DIR / "20251204-수요개발_신규계량기사용량현황.xlsx",
}

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

# KPI/요약표용 계량기 수(사용자 고정값)
TOTAL_METERS_NO_APT_FIXED = 2_891
TOTAL_METERS_INCL_APT_FIXED = 17_745
HOME_METERS_FIXED = 2_187
NONRES_METERS_FIXED = 704

# 상단 KPI용 전체 시공업체 수(1종) 고정값
TOTAL_COMPANY_FIXED = 70

# --------------------------------------------------
# 유틸 함수
# --------------------------------------------------
def fmt_int(x: float) -> str:
    """정수 + 천단위 콤마"""
    return f"{int(round(x)):,}"


def get_month_cols(df: pd.DataFrame) -> List:
    """연월(YYYYMM) 숫자형 컬럼만 추출"""
    return [c for c in df.columns if isinstance(c, (int, np.integer))]


def build_detached_avg_by_col(month_cols: List[int]) -> Dict[int, float]:
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
         - 우선 원본 컬럼 '연간 예상사용량' 이 있으면 그 값을 사용
         - 없으면 월별 값 중 숫자가 있는 달만 골라 평균(= 합계 / 값이 있는 달 수)
         - 월평균 × 12개월 → 연간사용량_추정
    """
    df = df_raw.copy()

    # 1종 시공업체만 사용
    if "업종" in df.columns:
        df = df[df["업종"] == "가스시공업 제1종"].copy()

    month_cols = get_month_cols(df)
    detached_avg_by_col = build_detached_avg_by_col(month_cols)

    # 아파트 제외
    if "자체업종명" in df.columns:
        df = df[df["자체업종명"] != "아파트"].copy()

    # 연립/다세대 -> 단독주택
    if "자체업종명" in df.columns and "용도" in df.columns:
        mask_multi = df["자체업종명"].isin(["연립주택", "다세대주택"])
        df.loc[mask_multi, "용도"] = "단독주택"

    # 사용여부 'Y' 만 사용 (있으면 적용)
    if "사용여부" in df.columns:
        df = df[df["사용여부"] == "Y"].copy()

    # 문자열/콤마가 섞인 '연간 예상사용량' 숫자 변환
    annual_col = None
    for cand in ["연간 예상사용량", "연간예상사용량"]:
        if cand in df.columns:
            annual_col = cand
            break
    if annual_col is not None:
        df[annual_col] = (
            df[annual_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"": np.nan})
        )
        df[annual_col] = pd.to_numeric(df[annual_col], errors="coerce")

    # 계량기별 연간 사용량 추정
    def compute_annual(row):
        # 월별 사용량
        usage = row[month_cols].astype(float) if month_cols else pd.Series(dtype=float)

        # 단독주택: 월별 보정 + 합계
        if "용도" in row and row["용도"] == "단독주택":
            for col in month_cols:
                base = detached_avg_by_col.get(col)
                v = usage[col]
                # 빈칸(NaN) 또는 0 → 단독주택 월평균으로 강제 치환
                if pd.isna(v) or v == 0:
                    if not pd.isna(base):
                        usage[col] = base
            return float(usage.sum())

        # 가정용외: 우선 '연간 예상사용량' 사용, 없으면 월평균×12
        if annual_col is not None:
            base_annual = row.get(annual_col, np.nan)
            if not pd.isna(base_annual):
                return float(base_annual)

        # 백업 로직: 월평균 × 12개월
        vals = usage.dropna()
        if len(vals) == 0:
            return 0.0
        monthly_avg = float(vals.mean())  # 예: 3달 값 있으면 /3
        return monthly_avg * 12.0        # 월평균 × 12개월

    df["연간사용량_추정"] = df.apply(compute_annual, axis=1)

    # 대분류(설명용): 가정용 vs 가정용외
    if "용도" in df.columns:
        df["대분류"] = np.where(df["용도"] == "단독주택", "가정용(단독주택)", "가정용외")
    else:
        df["대분류"] = "가정용외"

    # 시공업체별 집계 (전체 기준)
    if "시공업체" not in df.columns:
        df["시공업체"] = "미상"

    if "계량기번호" not in df.columns:
        df["계량기번호"] = df.index.astype(str)

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


@st.cache_data
def load_yearly_dataset() -> Tuple[Dict[int, Dict[str, pd.DataFrame]], List[int]]:
    """
    연간분석용: YEARLY_FILES에 등록된 연도별 파일을 모두 전처리해서 반환
    반환 형식:
      data_by_year[연도] = {
         "df_proc": ...,
         "agg_all": ...,
         "eligible": ...,
         "usage_by_type": ...,
         "usage_by_type_nonres": ...,
      }
    """
    data_by_year: Dict[int, Dict[str, pd.DataFrame]] = {}
    years: List[int] = []

    for year, path in YEARLY_FILES.items():
        if path.exists():
            raw = pd.read_excel(path)
            df_proc, agg_all, eligible, usage_by_type, usage_by_type_nonres, _ = preprocess(raw)
            data_by_year[year] = {
                "df_proc": df_proc,
                "agg_all": agg_all,
                "eligible": eligible,
                "usage_by_type": usage_by_type,
                "usage_by_type_nonres": usage_by_type_nonres,
            }
            years.append(year)

    years = sorted(years)
    return data_by_year, years


# --------------------------------------------------
# (이전 업로드 기반 평가점수 관련 함수는 그대로 두지만, 현재 버전에서는 사용하지 않음)
# --------------------------------------------------
def find_eval_sheet(xls: pd.ExcelFile) -> str | None:
    for sheet in xls.sheet_names:
        df_tmp = xls.parse(sheet)
        cols = set(map(str, df_tmp.columns))
        if {"구분", "총점"}.issubset(cols):
            return sheet
    return None


def load_eval_scores(file) -> pd.DataFrame | None:
    xls = pd.ExcelFile(file)
    sheet = find_eval_sheet(xls)
    if sheet is None:
        return None

    df = xls.parse(sheet)
    base_cols = ["구분", "총점"]
    for c in base_cols:
        if c not in df.columns:
            return None

    extra_col = None
    for c in df.columns:
        s = str(c)
        if "2-3" in s or "기존" in s:
            extra_col = c
            break

    cols = base_cols.copy()
    if extra_col is not None:
        cols.append(extra_col)

    df = df[cols].copy()
    df = df.dropna(subset=["구분"])
    df["총점"] = pd.to_numeric(df["총점"], errors="coerce").fillna(0)

    if extra_col is not None:
        df[extra_col] = pd.to_numeric(df[extra_col], errors="coerce").fillna(0)
    else:
        df[extra_col] = 0

    df = df.rename(columns={extra_col: "기존주택점수"})
    return df


# --------------------------------------------------
# 메인 타이틀 & 기본 설명
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

# --------------------------------------------------
# 상단 KPI
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    # 70개 고정
    st.metric("전체 시공업체 수 (1종)", f"{TOTAL_COMPANY_FIXED:,} 개")
with col2:
    st.metric("포상 기준 충족 업체 수", f"{eligible.shape[0]:,} 개")
with col3:
    st.metric(
        "전체 신규계량기 수 (공동주택 제외)",
        f"{TOTAL_METERS_NO_APT_FIXED:,} 전",
    )
with col4:
    st.metric(
        "전체 신규계량기 수 (공동주택 포함)",
        f"{TOTAL_METERS_INCL_APT_FIXED:,} 전",
    )

# --------------------------------------------------
# 탭 구성
# --------------------------------------------------
tab_rank, tab_type, tab_detail, tab_final, tab_yearly = st.tabs(
    ["업체별 순위", "용도별 분석", "업체별 용도 분석", "최종분석", "연간분석"]
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
            "계량기 수(전)": HOME_METERS_FIXED,   # 고정값
            "추정 연간사용량(m³)": df_home["연간사용량_추정"].sum(),
        },
        {
            "대분류": "가정용외",
            "계량기 수(전)": NONRES_METERS_FIXED,  # 고정값
            "추정 연간사용량(m³)": df_nonres_rows["연간사용량_추정"].sum(),
        },
        {
            "대분류": "합계",
            "계량기 수(전)": TOTAL_METERS_NO_APT_FIXED,  # 고정값
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
# 탭 4 : 최종분석  (종합점수 고정표 + 포상 표시)
# --------------------------------------------------
with tab_final:
    st.subheader("※ 최종분석 - 종합점수 기반 포상 추천")

    st.markdown(
        """
- 별도 업로드 없이 **평가점수표에서 산정한 최종 점수**를 그대로 사용합니다.  
- 2-3항목(기존주택 비율)은 이미 `수요개발관리` 점수 안에 반영되어 있으므로  
  이 탭에서는 별도 비율 계산 없이 **총점 기준 순위**만 사용합니다.
"""
    )

    # 네가 올려준 엑셀/이미지 기준 최종 점수표 (고정값)
    eval_data = [
        {"순번": 1, "업체": "보민에너지(주)",        "경영일반": 3, "수요개발관리": 34, "품질관리": 41, "감점": 0, "총점": 78, "순위": 1},
        {"순번": 2, "업체": "(주)대경지엔에스",      "경영일반": 3, "수요개발관리": 18, "품질관리": 45, "감점": 0, "총점": 66, "순위": 2},
        {"순번": 3, "업체": "주식회사 유성산업개발", "경영일반": 3, "수요개발관리": 26, "품질관리": 37, "감점": 0, "총점": 66, "순위": 2},
        {"순번": 4, "업체": "(주)영화이엔지",        "경영일반": 4, "수요개발관리": 14, "품질관리": 43, "감점": 0, "총점": 61, "순위": 4},
        {"순번": 5, "업체": "디에스이앤씨(주)",      "경영일반": 5, "수요개발관리": 34, "품질관리": 16, "감점": 0, "총점": 55, "순위": 5},
        {"순번": 6, "업체": "주식회사삼주이엔지",    "경영일반": 4, "수요개발관리": 16, "품질관리": 30, "감점": 0, "총점": 50, "순위": 6},
        {"순번": 7, "업체": "(주)신한설비",          "경영일반": 4, "수요개발관리": 18, "품질관리": 17, "감점": 0, "총점": 39, "순위": 7},
        {"순번": 8, "업체": "동우에너지주식회사",    "경영일반": 2, "수요개발관리": 14, "품질관리": 23, "감점": 0, "총점": 39, "순위": 7},
        {"순번": 9, "업체": "금강에너지 주식회사",   "경영일반": 2, "수요개발관리": 14, "품질관리": 23, "감점": 0, "총점": 39, "순위": 7},
    ]
    eval_df = pd.DataFrame(eval_data)

    # 1위 업체에 '포상' 표기
    eval_df["포상"] = ""
    eval_df.loc[eval_df["순위"] == 1, "포상"] = "포상"

    def highlight_awards(row):
        if row["순위"] == 1:
            return ["background-color: #FFF4CC" for _ in row]
        return [""] * len(row)

    disp_eval = eval_df[
        ["순번", "업체", "경영일반", "수요개발관리", "품질관리", "감점", "총점", "순위", "포상"]
    ]
    styled_eval = center_style(disp_eval, highlight_awards)

    st.dataframe(
        styled_eval,
        use_container_width=True,
        hide_index=True,
        column_config={
            "순번": st.column_config.Column("순번", width="small"),
            "순위": st.column_config.Column("순위", width="small"),
        },
    )

    st.caption("- 노란색 행이 **포상 대상(1위 업체)** 이고, `포상` 컬럼에 표시됩니다.")


# --------------------------------------------------
# 탭 5 : 연간분석 (연도별 추이)
# --------------------------------------------------
with tab_yearly:
    st.subheader("📆 연간 추이 분석")

    data_by_year, years = load_yearly_dataset()

    if not years:
        st.info(
            "연간 분석에 사용할 연도별 파일을 찾지 못했어. "
            "폴더 안에 '2023~2025 수요개발_신규공급계량기사용량현황.xlsx' 파일이 있는지 확인해줘."
        )
    else:
        st.markdown(
            f"- 분석 대상 연도: **{', '.join(map(str, years))}년**  "
        )

        sub1, sub2, sub3, sub4 = st.tabs(
            [
                "연도별 포상대상 현황",
                "연도별 용도 패턴",
                "업체별 연간 실적 추이",
                "연도별 Top-N 업체",
            ]
        )

        # ─────────────────────────────────────────────
        # 서브탭1: 연도별 포상대상 현황
        # ─────────────────────────────────────────────
        with sub1:
            st.markdown("#### 🏆 연도별 포상 기준 충족 현황")

            rows = []
            for y in years:
                info = data_by_year[y]
                agg_y = info["agg_all"]
                eligible_y = info["eligible"]

                total_comp = agg_y.shape[0]
                eligible_cnt = eligible_y.shape[0]
                total_meters = agg_y["신규계량기수"].sum()
                total_usage = agg_y["연간사용량합계"].sum()

                rows.append(
                    {
                        "연도": y,
                        "전체 시공업체 수(1종)": total_comp,
                        "포상 기준 충족 업체 수": eligible_cnt,
                        "포상 기준 충족 비율(%)": eligible_cnt / total_comp * 100
                        if total_comp > 0
                        else 0,
                        "전체 신규계량기 수(전)": total_meters,
                        "추정 연간사용량 합계(m³)": total_usage,
                    }
                )

            year_summary = pd.DataFrame(rows)
            disp = year_summary.copy()
            disp["전체 신규계량기 수(전)"] = disp["전체 신규계량기 수(전)"].map(fmt_int)
            disp["추정 연간사용량 합계(m³)"] = disp["추정 연간사용량 합계(m³)"].map(
                fmt_int
            )
            disp["포상 기준 충족 비율(%)"] = disp["포상 기준 충족 비율(%)"].map(
                lambda x: f"{x:,.1f}%"
            )

            styled_year = center_style(disp)
            st.dataframe(
                styled_year,
                use_container_width=True,
                hide_index=True,
            )

            # 포상대상 업체수/비율 라인차트
            fig_line1 = px.line(
                year_summary,
                x="연도",
                y=["전체 시공업체 수(1종)", "포상 기준 충족 업체 수"],
                markers=True,
            )
            fig_line1.update_layout(
                yaxis_title="업체 수(개)",
                legend_title="구분",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_line1, use_container_width=True)

            fig_line2 = px.line(
                year_summary,
                x="연도",
                y="포상 기준 충족 비율(%)",
                markers=True,
            )
            fig_line2.update_layout(
                yaxis_title="포상 기준 충족 비율(%)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_line2, use_container_width=True)

        # ─────────────────────────────────────────────
        # 서브탭2: 연도별 용도 패턴
        # ─────────────────────────────────────────────
        with sub2:
            st.markdown("#### 🔍 연도별 용도별 사용량 패턴 변화")

            rows_cat = []
            rows_type = []

            for y in years:
                info = data_by_year[y]
                df_y = info["df_proc"]

                df_home_y = df_y[df_y["용도"] == "단독주택"].copy()
                df_nonres_y = df_y[
                    (df_y["용도"] != "단독주택") & (df_y["용도"] != "공동주택")
                ].copy()

                rows_cat.append(
                    {
                        "연도": y,
                        "대분류": "가정용(공동주택 제외)",
                        "연간사용량(m³)": df_home_y["연간사용량_추정"].sum(),
                    }
                )
                rows_cat.append(
                    {
                        "연도": y,
                        "대분류": "가정용외",
                        "연간사용량(m³)": df_nonres_y["연간사용량_추정"].sum(),
                    }
                )

                # 세부 용도별 (가정용외 중심)
                usage_type_y = (
                    df_nonres_y.groupby("용도")["연간사용량_추정"]
                    .sum()
                    .reset_index()
                )
                usage_type_y["연도"] = y
                rows_type.append(usage_type_y)

            cat_df = pd.DataFrame(rows_cat)
            type_df = pd.concat(rows_type, ignore_index=True) if rows_type else None

            # 대분류 패턴 (가정용 vs 가정용외)
            fig_cat = px.bar(
                cat_df,
                x="연도",
                y="연간사용량(m³)",
                color="대분류",
                barmode="group",
                text="연간사용량(m³)",
            )
            fig_cat.update_traces(
                texttemplate="%{text:,.0f}", textposition="outside"
            )
            fig_cat.update_layout(
                yaxis_title="연간사용량(m³)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            st.markdown("---")
            st.markdown("##### 📎 가정용외 세부 용도별 추세")

            if type_df is None or type_df.empty:
                st.info("가정용외 세부 용도 데이터가 없습니다.")
            else:
                top_types = (
                    type_df.groupby("용도")["연간사용량_추정"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .index.tolist()
                )
                selected_types = st.multiselect(
                    "비교할 용도 선택 (최대 5개 정도 추천)", top_types, default=top_types[:3]
                )

                if selected_types:
                    sub_type = type_df[type_df["용도"].isin(selected_types)].copy()
                    fig_type_year = px.line(
                        sub_type,
                        x="연도",
                        y="연간사용량_추정",
                        color="용도",
                        markers=True,
                    )
                    fig_type_year.update_layout(
                        yaxis_title="연간사용량(m³)",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_type_year, use_container_width=True)

        # ─────────────────────────────────────────────
        # 서브탭3: 업체별 연간 실적 추이
        # ─────────────────────────────────────────────
        with sub3:
            st.markdown("#### 🏗 업체별 연간 실적 추이")

            # 모든 연도에 등장한 업체 리스트
            company_set = set()
            for y in years:
                company_set.update(
                    data_by_year[y]["agg_all"].index.tolist()
                )
            company_list = sorted(company_set)

            selected_company = st.selectbox(
                "시공업체 선택", company_list
            )

            rows_comp = []
            for y in years:
                agg_y = data_by_year[y]["agg_all"]
                if selected_company in agg_y.index:
                    r = agg_y.loc[selected_company]
                    meters = r["신규계량기수"]
                    usage = r["연간사용량합계"]
                else:
                    meters = 0
                    usage = 0
                rows_comp.append(
                    {
                        "연도": y,
                        "신규계량기 수(전)": meters,
                        "추정 연간사용량 합계(m³)": usage,
                    }
                )

            comp_trend = pd.DataFrame(rows_comp)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_m = px.line(
                    comp_trend,
                    x="연도",
                    y="신규계량기 수(전)",
                    markers=True,
                )
                fig_m.update_layout(
                    yaxis_title="신규계량기 수(전)",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_m, use_container_width=True)

            with col_b:
                fig_u = px.line(
                    comp_trend,
                    x="연도",
                    y="추정 연간사용량 합계(m³)",
                    markers=True,
                )
                fig_u.update_layout(
                    yaxis_title="추정 연간사용량 합계(m³)",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_u, use_container_width=True)

            st.dataframe(
                center_style(
                    comp_trend.assign(
                        **{
                            "신규계량기 수(전)": comp_trend["신규계량기 수(전)"].map(fmt_int),
                            "추정 연간사용량 합계(m³)": comp_trend[
                                "추정 연간사용량 합계(m³)"
                            ].map(fmt_int),
                        }
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )

        # ─────────────────────────────────────────────
        # 서브탭4: 연도별 Top-N 업체
        # ─────────────────────────────────────────────
        with sub4:
            st.markdown("#### 🌟 연도별 Top-N 포상 후보 비교")

            year_sel = st.selectbox("연도 선택", years, index=len(years) - 1)
            top_n = st.slider("Top-N 범위 선택", min_value=3, max_value=15, value=10)

            info_y = data_by_year[year_sel]
            agg_y = info_y["agg_all"].copy()
            agg_y = agg_y.sort_values("연간사용량합계", ascending=False).head(top_n)
            agg_y = agg_y.reset_index()

            agg_y["추정 연간사용량 합계(m³)"] = agg_y["연간사용량합계"].map(fmt_int)
            agg_y["신규계량기 수(전)"] = agg_y["신규계량기수"].map(fmt_int)
            agg_y["순위"] = np.arange(1, len(agg_y) + 1)

            disp_cols = [
                "순위",
                "시공업체",
                "신규계량기 수(전)",
                "추정 연간사용량 합계(m³)",
            ]
            st.dataframe(
                center_style(agg_y[disp_cols]),
                use_container_width=True,
                hide_index=True,
            )

            fig_top = px.bar(
                agg_y,
                x="시공업체",
                y="연간사용량합계",
                text="추정 연간사용량 합계(m³)",
            )
            fig_top.update_traces(textposition="outside")
            fig_top.update_layout(
                xaxis_title="시공업체",
                yaxis_title="추정 연간사용량 합계(m³)",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_top, use_container_width=True)
