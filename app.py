from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 페이지 기본 설정 (와이드 레이아웃, 장식용 이모지 미사용)
st.set_page_config(layout="wide", page_title="ETF 레버리지 및 분산투자 대시보드")


# ==========================================
# 1. 데이터 로딩 및 순수 계산 함수 (st. 호출 없음)
# ==========================================

@st.cache_data
def load_data():
    """data/etf_prices.csv를 읽어 datetime을 날짜로 변환하고 symbol을 열로 하는 종가 피벗 테이블을 반환한다."""
    csv_path = Path(__file__).parent / "data" / "etf_prices.csv"
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    pivot_df = df.pivot(index="datetime", columns="symbol", values="close").sort_index()
    return pivot_df


def calc_total_return(prices):
    """가격 시계열의 첫 종가 대비 마지막 종가의 총수익률을 계산한다."""
    return (prices.iloc[-1] / prices.iloc[0]) - 1.0


def calc_cagr(prices):
    """실제 경과일(경과일 / 365.25)을 기준으로 연환산복리수익률(CAGR)을 계산한다."""
    total_ret = calc_total_return(prices)
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return ((1.0 + total_ret) ** (1.0 / years)) - 1.0


def calc_volatility(prices):
    """일간수익률의 표본표준편차에 sqrt(252)를 곱해 연변동성을 계산한다."""
    daily_ret = prices.pct_change().dropna()
    return daily_ret.std(ddof=1) * (252.0 ** 0.5)


def calc_drawdown_series(prices):
    """각 시점까지의 누적 최고 종가 대비 현재 종가의 낙폭(Drawdown) 시계열을 계산한다."""
    cummax = prices.cummax()
    return (prices / cummax) - 1.0


def calc_mdd(prices):
    """낙폭 시계열의 최솟값인 최대낙폭(MDD)을 계산한다."""
    return calc_drawdown_series(prices).min()


def calc_metrics_summary(prices_df, symbols):
    """지정한 심볼 목록에 대해 총수익률, CAGR, 연변동성, MDD를 계산하여 리스트로 반환한다."""
    rows = []
    for sym in symbols:
        s = prices_df[sym]
        rows.append({
            "symbol": sym,
            "total_return": calc_total_return(s),
            "cagr": calc_cagr(s),
            "volatility": calc_volatility(s),
            "mdd": calc_mdd(s),
        })
    return rows


def normalize_prices(prices_df):
    """선택 구간 첫날의 종가를 100으로 맞춰 정규화한 가격 데이터프레임을 반환한다."""
    return (prices_df / prices_df.iloc[0]) * 100.0


def build_rebalanced_portfolio(prices_df, spy_weight_pct):
    """매일 지정한 비중(SPY 비중 %, 나머지 GLD)으로 리밸런싱하는 포트폴리오의 가격 지수(첫날=100)를 계산한다."""
    daily_ret = prices_df[["SPY", "GLD"]].pct_change().dropna()
    w_spy = spy_weight_pct / 100.0
    w_gld = 1.0 - w_spy
    port_ret = (w_spy * daily_ret["SPY"]) + (w_gld * daily_ret["GLD"])

    port_prices = pd.Series(100.0, index=prices_df.index)
    port_prices.iloc[1:] = 100.0 * (1.0 + port_ret).cumprod()
    return port_prices


def calc_allocation_grid(prices_df):
    """SPY 비중을 0%부터 100%까지 10% 간격으로 바꿨을 때의 총수익률과 연변동성 테이블을 반환한다."""
    records = []
    for w in range(0, 101, 10):
        p_series = build_rebalanced_portfolio(prices_df, w)
        records.append({
            "spy_weight": w,
            "gld_weight": 100 - w,
            "label": f"SPY {w}%",
            "total_return_pct": calc_total_return(p_series) * 100.0,
            "volatility_pct": calc_volatility(p_series) * 100.0,
            "mdd_pct": calc_mdd(p_series) * 100.0,
        })
    return pd.DataFrame(records)


# ==========================================
# 2. 화면 그리기 함수 (탭 1, 탭 2 분리)
# ==========================================

def render_tab1(sliced_prices, start_date, end_date):
    """탭 1 '레버리지의 함정' 화면을 그린다."""
    st.header("레버리지의 함정: 배수는 하루짜리 약속이지 5년짜리 약속이 아닙니다")

    lev_symbols = ["QQQ", "QLD", "TQQQ"]
    lev_prices = sliced_prices[lev_symbols]

    # 3-1 & 3-3. 누적 성과 곡선과 낙폭 그래프를 나란히 배치
    use_log = st.checkbox("누적 성과 곡선 세로축 로그 스케일 적용", value=False, key="tab1_log_toggle")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. 누적 성과 곡선 (시작일 = 100)")
        norm_df = normalize_prices(lev_prices)
        fig_norm = px.line(
            norm_df,
            x=norm_df.index,
            y=lev_symbols,
            log_y=use_log,
            labels={"value": "정규화 가격 (시작일=100)", "datetime": "날짜", "variable": "종목"},
        )
        fig_norm.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_title_text="종목")
        st.plotly_chart(fig_norm, width="stretch")

    with col_right:
        st.subheader("2. 낙폭(Drawdown) 추이")
        dd_df = calc_drawdown_series(lev_prices) * 100.0
        fig_dd = px.line(
            dd_df,
            x=dd_df.index,
            y=lev_symbols,
            labels={"value": "고점 대비 낙폭 (%)", "datetime": "날짜", "variable": "종목"},
        )
        fig_dd.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_title_text="종목")
        st.plotly_chart(fig_dd, width="stretch")

    # 3-2. 성과 요약표
    st.subheader("3. 성과 요약표")
    raw_metrics = calc_metrics_summary(lev_prices, lev_symbols)
    role_map = {"QQQ": "나스닥100 (1배)", "QLD": "나스닥100 (2배)", "TQQQ": "나스닥100 (3배)"}
    display_rows = []
    for m in raw_metrics:
        display_rows.append({
            "심볼": m["symbol"],
            "구분": role_map[m["symbol"]],
            "총수익률": f"{m['total_return']*100:+.1f}%",
            "연환산수익률(CAGR)": f"{m['cagr']*100:.1f}%",
            "연변동성": f"{m['volatility']*100:.1f}%",
            "최대낙폭(MDD)": f"{m['mdd']*100:.1f}%",
        })
    st.table(pd.DataFrame(display_rows))

    st.markdown(
        "- **총수익률**: 선택 구간 첫날 종가 대비 마지막 날 종가의 누적 변화율입니다.\n"
        "- **연환산수익률(CAGR)**: 전체 기간의 성과를 연 단위 복리 성장률로 환산한 값입니다.\n"
        "- **연변동성**: 일간수익률이 얼마나 크게 흔들렸는지를 연 단위로 환산한 위험 지표입니다.\n"
        "- **최대낙폭(MDD)**: 선택한 기간 중 **고점 대비 얼마나 빠졌는가**(가장 깊게 하락한 비율)를 뜻합니다."
    )

    # 3-4. 원금 회복 계산기
    st.subheader("4. 원금 회복 계산기")
    drop_pct = st.slider("계좌 하락률 선택 (%)", min_value=0, max_value=90, value=50, step=5)
    drop_ratio = drop_pct / 100.0
    req_gain_pct = ((1.0 / (1.0 - drop_ratio)) - 1.0) * 100.0

    st.metric(
        label=f"-{drop_pct}% 하락 후 원금 회복에 필요한 상승률",
        value=f"+{req_gain_pct:.1f}%",
    )
    st.markdown("**-50% 하락은 +100% 상승이 있어야 원금입니다. 하락과 상승은 대칭이 아닙니다.**")

    # 3-5. 핵심 메시지 자동 출력
    st.subheader("5. 선택 구간 핵심 요약")
    metrics_by_sym = {m["symbol"]: m for m in raw_metrics}
    qqq_ret = metrics_by_sym["QQQ"]["total_return"]
    tqqq_ret = metrics_by_sym["TQQQ"]["total_return"]
    tqqq_mdd = metrics_by_sym["TQQQ"]["mdd"]
    years_span = (lev_prices.index[-1] - lev_prices.index[0]).days / 365.25

    if qqq_ret > 0:
        ratio = tqqq_ret / qqq_ret
        if ratio < 1.0:
            ratio_sentence = (
                f"배수는 3배지만 약 {years_span:.1f}년 수익률은 1배인 QQQ보다도 낮았습니다({ratio:.1f}배)."
            )
        elif ratio < 3.0:
            ratio_sentence = (
                f"배수는 3배지만 약 {years_span:.1f}년 수익률 차이는 {ratio:.1f}배에 그쳤습니다."
            )
        else:
            ratio_sentence = (
                f"상승장이 지속되어 약 {years_span:.1f}년 수익률 차이가 {ratio:.1f}배로 3배를 넘어섰습니다."
            )
    else:
        ratio_sentence = "이 구간에서는 기초지수(QQQ) 자체가 하락하면서 3배 레버리지의 손실 폭이 크게 확대되었습니다."

    auto_msg = (
        f"{start_date} ~ {end_date} 구간에서 TQQQ(3배)는 {tqqq_ret*100:+.1f}%, "
        f"QQQ(1배)는 {qqq_ret*100:+.1f}%였습니다.\n\n"
        f"{ratio_sentence}\n\n"
        f"그동안 TQQQ의 최대낙폭은 {tqqq_mdd*100:.1f}%였습니다."
    )
    st.info(auto_msg)

    # 3-6. 시작일 변경 유도 안내
    st.markdown("---")
    st.markdown("**사이드바에서 분석 시작일을 2023-01-03으로 바꿔 보십시오. 결론이 뒤집힙니다.**")


def render_tab2(sliced_prices):
    """탭 2 '분산투자의 힘' 화면을 그린다."""
    st.header("분산투자의 힘: 변동성이 큰 자산을 섞었는데 포트폴리오 변동성이 오히려 줄어듭니다")

    all_symbols = ["SPY", "VTI", "QQQ", "QLD", "TQQQ", "SCHD", "GLD"]
    daily_returns = sliced_prices[all_symbols].pct_change().dropna()

    # 4-1. 상관계수 히트맵
    st.subheader("1. 7종목 일간수익률 상관계수 히트맵")
    corr_matrix = daily_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        zmin=-1.0,
        zmax=1.0,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="상관계수"),
    )
    fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_corr, width="stretch")

    # 4-2. 히트맵 해석 문장
    c_spy_vti = corr_matrix.loc["SPY", "VTI"]
    c_qqq_qld = corr_matrix.loc["QQQ", "QLD"]
    c_qqq_tqqq = corr_matrix.loc["QQQ", "TQQQ"]
    c_gld_spy = corr_matrix.loc["GLD", "SPY"]
    c_gld_schd = corr_matrix.loc["GLD", "SCHD"]

    st.markdown(
        f"- **SPY와 VTI의 상관계수는 {c_spy_vti:.2f}로 1에 매우 가깝습니다**: "
        f"이름만 다를 뿐 거의 같은 방향으로 움직이므로 **둘 다 보유하는 것은 분산이 아닙니다.**\n"
        f"- **QQQ, QLD, TQQQ는 사실상 같은 자산입니다**: "
        f"QQQ-QLD({c_qqq_qld:.2f}), QQQ-TQQQ({c_qqq_tqqq:.2f}) 모두 상관계수가 1.00에 수렴하여 섞어 담아도 분산 효과가 전혀 없습니다.\n"
        f"- **GLD(금)만 다른 종목들과 상관이 낮습니다**: "
        f"GLD-SPY({c_gld_spy:.2f}), GLD-SCHD({c_gld_schd:.2f})처럼 상관관계가 낮은 자산을 섞을 때 진정한 분산 효과가 나옵니다."
    )

    # 4-3. 두 자산 배분 실험
    st.subheader("2. 두 자산(SPY + GLD) 배분 실험")
    st.caption("가정: 포트폴리오 일간수익률 = SPY비중 × SPY수익률 + GLD비중 × GLD수익률 (매일 비중을 다시 맞춘다는 일간 리밸런싱 가정)")

    spy_w = st.slider("SPY(미국 대형주) 비중 (%)", min_value=0, max_value=100, value=50, step=5)
    gld_w = 100 - spy_w
    st.write(f"선택한 비중: **SPY {spy_w}% + GLD {gld_w}%**")

    port_prices = build_rebalanced_portfolio(sliced_prices, spy_w)
    spy_only_prices = sliced_prices["SPY"]

    port_tot = calc_total_return(port_prices)
    port_vol = calc_volatility(port_prices)
    port_mdd = calc_mdd(port_prices)

    spy_tot = calc_total_return(spy_only_prices)
    spy_vol = calc_volatility(spy_only_prices)
    spy_mdd = calc_mdd(spy_only_prices)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="포트폴리오 총수익률",
            value=f"{port_tot*100:+.1f}%",
            delta=f"{(port_tot - spy_tot)*100:+.1f}%p (SPY 100% 대비)",
        )
    with m2:
        st.metric(
            label="포트폴리오 연변동성",
            value=f"{port_vol*100:.1f}%",
            delta=f"{(port_vol - spy_vol)*100:+.1f}%p (SPY 100% 대비)",
            delta_color="inverse",
        )
    with m3:
        st.metric(
            label="포트폴리오 최대낙폭(MDD)",
            value=f"{port_mdd*100:.1f}%",
            delta=f"{(port_mdd - spy_mdd)*100:+.1f}%p (SPY 100% 대비)",
        )

    # 4-4. 비중별 위험-수익 산점도
    st.subheader("3. 비중별 위험-수익 산점도 (SPY 0% ~ 100%, 10% 간격)")
    grid_df = calc_allocation_grid(sliced_prices)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=grid_df["volatility_pct"],
            y=grid_df["total_return_pct"],
            mode="lines+markers+text",
            text=grid_df["label"],
            textposition="top center",
            marker=dict(size=10, color="#1f77b4"),
            line=dict(color="#aec7e8", dash="dot"),
            name="10% 간격 비중 조합",
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[port_vol * 100.0],
            y=[port_tot * 100.0],
            mode="markers",
            marker=dict(size=16, color="#d62728", line=dict(width=2, color="black")),
            name=f"현재 선택 (SPY {spy_w}% / GLD {gld_w}%)",
        )
    )
    fig_scatter.update_layout(
        xaxis_title="연변동성 (%)",
        yaxis_title="총수익률 (%)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_scatter, width="stretch")

    min_vol_row = grid_df.loc[grid_df["volatility_pct"].idxmin()]
    gld_100_vol = grid_df.loc[grid_df["spy_weight"] == 0, "volatility_pct"].iloc[0]
    spy_100_vol = grid_df.loc[grid_df["spy_weight"] == 100, "volatility_pct"].iloc[0]

    st.info(
        f"이 그래프의 왼쪽 끝(연변동성 최저 지점)은 100% 주식(SPY 100%, {spy_100_vol:.1f}%)도 "
        f"100% 금(GLD 100%, {gld_100_vol:.1f}%)도 아닌 "
        f"**SPY {int(min_vol_row['spy_weight'])}% / GLD {int(min_vol_row['gld_weight'])}% "
        f"(연변동성 {min_vol_row['volatility_pct']:.1f}%)** 입니다. "
        f"상관이 낮은 두 자산을 섞은 쪽이 각각을 단독으로 보유할 때보다 덜 흔들립니다."
    )

    # 4-5. 자산군 성과 비교
    st.subheader("4. 자산군 누적 성과 비교 (SPY, GLD, SCHD)")
    asset_symbols = ["SPY", "GLD", "SCHD"]
    norm_assets = normalize_prices(sliced_prices[asset_symbols])

    fig_assets = px.line(
        norm_assets,
        x=norm_assets.index,
        y=asset_symbols,
        labels={"value": "정규화 가격 (시작일=100)", "datetime": "날짜", "variable": "종목"},
    )
    fig_assets.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_title_text="종목")
    st.plotly_chart(fig_assets, width="stretch")
    st.caption(
        "주석: SCHD는 배당 수익이 핵심인 배당성장 ETF이나, 본 데이터는 배당이 제외된 종가(close) 기준이므로 실제 총수익률 대비 과소평가되어 있습니다."
    )


# ==========================================
# 3. 메인 엔트리포인트 및 사이드바 구성
# ==========================================

def main():
    prices_data = load_data()

    min_dt = prices_data.index.min().date()
    max_dt = prices_data.index.max().date()

    st.sidebar.header("분석 기간 설정")
    start_date = st.sidebar.date_input(
        "분석 시작일",
        value=pd.to_datetime("2021-08-23").date(),
        min_value=min_dt,
        max_value=max_dt,
    )
    end_date = st.sidebar.date_input(
        "분석 종료일",
        value=pd.to_datetime("2026-08-20").date(),
        min_value=min_dt,
        max_value=max_dt,
    )

    # 사이드바 하단 배당 미포함 주의 문구 상시 표시 (접거나 숨기지 않음)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "주의: 종가 기준 가격 수익률이며 배당은 포함되지 않았습니다. 실제 성과와 다릅니다."
    )

    # 시작일 > 종료일 검증
    if start_date > end_date:
        st.error("분석 시작일이 종료일보다 늦습니다. 시작일을 종료일 이전으로 설정해 주십시오.")
        st.stop()

    sliced_prices = prices_data.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

    if len(sliced_prices) < 2:
        st.error("선택한 구간에 영업일 데이터가 부족합니다. 기간을 넓혀 주십시오.")
        st.stop()

    # 영업일 수 60일 미만 경고 (계산은 계속 진행)
    if len(sliced_prices) < 60:
        st.warning("구간이 짧아 연환산 지표가 불안정합니다")

    st.title("미국 ETF 레버리지·분산투자 분석 대시보드")

    tab1, tab2 = st.tabs(["레버리지의 함정", "분산투자의 힘"])
    with tab1:
        render_tab1(sliced_prices, start_date, end_date)
    with tab2:
        render_tab2(sliced_prices)


if __name__ == "__main__":
    main()
