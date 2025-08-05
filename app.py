import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
import gspread
from gspread_dataframe import get_as_dataframe
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="ROgicX 작전 본부 v8.4", page_icon="🤖", layout="wide")

# ==============================================================================
# --- v8.4 안정화 유틸리티 함수 ---
# ==============================================================================
def safe_get_data(ticker, period="1y"):
    """안정적으로 yfinance 데이터를 로드합니다. 실패 시 None을 반환합니다."""
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        return None

def macd_crossover(macd_line, signal_line):
    """MACD 골든크로스 발생 여부를 명확하게 확인합니다."""
    if len(macd_line) < 3 or len(signal_line) < 3:
        return False
    return ((macd_line.iloc[-3] < signal_line.iloc[-3] and macd_line.iloc[-1] > signal_line.iloc[-1]) or
            (macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]))

# ==============================================================================
# --- 모든 계산 함수 ---
# ==============================================================================
def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(100)

def calculate_volume_ratio(volume_series, window=20):
    if len(volume_series) < window: return 1.0
    avg_volume = volume_series.rolling(window=window).mean().iloc[-1]
    last_volume = volume_series.iloc[-1]
    return (last_volume / avg_volume) if avg_volume > 1e-6 else 1.0

def get_price_change_rate(close_prices, window=3):
    if len(close_prices) < window:
        return 0
    return (close_prices.iloc[-1] / close_prices.iloc[-window] - 1) * 100

def calculate_bb_deviation(close_prices, window=20, num_std=2):
    ma = close_prices.rolling(window=window).mean()
    std = close_prices.rolling(window=window).std()
    lower_band = ma - (num_std * std)
    last_price = close_prices.iloc[-1]
    last_lower_band = lower_band.iloc[-1]
    if last_lower_band == 0: return 0.0
    deviation = ((last_price / last_lower_band) - 1) * 100
    return deviation

# ==============================================================================
# --- v8.4 핵심 분석 모듈 (최종 안정화 버전) ---
# ==============================================================================
def analyze_stock_v8_4(stock_info, tier, params):
    if not stock_info: return None

    deviation = stock_info.get('deviation', 0)
    bb_lower_dev = stock_info.get('bb_lower_dev', 0)
    decline_from_peak = stock_info.get('decline_from_peak', 0)
    rsi = stock_info.get('rsi', 50)
    macd_cross = stock_info.get('macd_cross', False)
    macd_latest = stock_info.get('macd_latest', 0)
    signal_latest = stock_info.get('signal_latest', 0)
    volume_ratio = stock_info.get('volume_ratio', 0)
    price_change_rate = get_price_change_rate(stock_info['close_prices'], window=3)

    tier_key = tier.replace('Tier ', '').replace('.', '_') if tier else '4'
    if f'tier{tier_key}_bb_dev' not in params: tier_key = '4'

    price_attractive_bb = (bb_lower_dev <= params[f'tier{tier_key}_bb_dev'])
    price_attractive_ma = (deviation <= params[f'tier{tier_key}_ma_dev'])
    price_attractive_peak = (decline_from_peak <= params[f'tier{tier_key}_peak_decline'])
    price_attractive = price_attractive_bb or price_attractive_ma or price_attractive_peak

    energy_condensed = (rsi <= params[f'tier{tier_key}_rsi'])
    market_agreed = (volume_ratio >= params[f'tier{tier_key}_vol'])

    is_watching = False
    is_captured = False
    if tier in ['Tier 1', 'Tier 4']:
        is_watching = price_attractive and energy_condensed
        if is_watching and market_agreed: is_captured = True
    elif tier in ['Tier 1.5', 'Tier 2']:
        is_watching = price_attractive and energy_condensed and macd_cross
        if is_watching and market_agreed: is_captured = True

    price_desc_parts = []
    if price_attractive_bb: price_desc_parts.append(f"BB({bb_lower_dev:.1f}%)")
    if price_attractive_ma: price_desc_parts.append(f"MA({deviation:.1f}%)")
    if price_attractive_peak: price_desc_parts.append(f"고점({decline_from_peak:.1f}%)")
    
    if price_attractive:
        price_text = " ".join(price_desc_parts)
    else:
        price_text = f"미달 (BB:{bb_lower_dev:.1f}%, MA:{deviation:.1f}%, 고점:{decline_from_peak:.1f}%)"
    
    energy_desc = "과매도" if rsi <= 35 else "과열" if rsi >= 65 else "중립"
    energy_text = f"{energy_desc} (RSI:{rsi:.1f})"

    trend_text = "상승 전환" if macd_cross else ("상승 추세" if macd_latest > signal_latest else "하락 추세")
    volume_desc = "급증" if volume_ratio >= 1.5 else "부족" if volume_ratio < 1.0 else "평균"
    volume_text = f"{volume_desc} ({volume_ratio:.1f}배)"

    status, status_order = "⚪️ 안정", 4
    if is_watching: status, status_order = "🟡 감시", 2
    if is_captured: status, status_order = "🟢 포착", 1

    if price_change_rate <= -7 and status == "⚪️ 안정": status, status_order = "⚡ 변동성", 3
    if price_change_rate >= 7: status, status_order = "⚠️ 과열", 5

    return {
        '상태': status, '종목명': stock_info['name'], '티어': tier,
        '3일 변동률': f"{price_change_rate:.1f}%",
        '가격 매력도': f"{'✅' if price_attractive else '❌'} ({price_text})",
        '에너지 응축': f"{'✅' if energy_condensed else '❌'} ({energy_text})",
        '추세 전환': f"{'✅' if macd_cross else '❌'} ({trend_text})",
        '시장 동의': f"{'✅' if market_agreed else '❌'} ({volume_text})",
        'status_order': status_order
    }

# --- 데이터 로딩 함수 ---
@st.cache_data(ttl=600)
def load_data_from_gsheet():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        SPREADSHEET_KEY = '1AG2QrAlcjksI2CWp_6IuL5jCrFhzpOGl7casHvFGvi8'
        spreadsheet = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = spreadsheet.get_worksheet(0)
        df = get_as_dataframe(worksheet, evaluate_formulas=True).dropna(how='all').dropna(axis=1, how='all')
        return df
    except Exception as e:
        st.error(f"Google Sheets 데이터를 불러오는 데 실패했습니다: {e}")
        return None

@st.cache_data(ttl=600)
def get_market_health_data():
    data = {}
    try:
        sp500 = safe_get_data('^GSPC', period='3mo')
        if sp500 is not None:
            data['sp500_close'] = sp500['Close'].iloc[-1]
            data['sp500_ma20'] = sp500['Close'].rolling(window=20).mean().iloc[-1]
            data['sp500_rsi'] = calculate_rsi(sp500['Close']).iloc[-1]

        vix_data = safe_get_data('^VIX', period='1d')
        if vix_data is not None: data['vix'] = vix_data['Close'][0]
            
        tnx_data = safe_get_data('^TNX', period='5d')
        if tnx_data is not None and len(tnx_data) >= 2:
            data['tnx_change'] = (tnx_data['Close'].iloc[-1] / tnx_data['Close'].iloc[-2] - 1) * 100
        else: data['tnx_change'] = 0

        fng_response = requests.get("https://api.alternative.me/fng/?limit=1")
        data['fng_value'] = int(fng_response.json()['data'][0]['value'])
    except Exception: pass
    return data

@st.cache_data
def get_stock_data(tickers, stock_names):
    stock_data = {}
    ticker_to_name = dict(zip(tickers, stock_names))
    for ticker in [t for t in tickers if t and isinstance(t, str) and t != 'CASH']:
        hist = safe_get_data(ticker, period='1y')
        if hist is not None and len(hist) > 63:
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()
            
            peak_3m = hist['Close'].rolling(window=63).max().iloc[-1]
            decline_from_peak = ((hist['Close'].iloc[-1] / peak_3m) - 1) * 100 if peak_3m > 0 else 0

            stock_data[ticker] = {
                'name': ticker_to_name.get(ticker, ticker),
                'deviation': ((hist['Close'].iloc[-1] / hist['Close'].rolling(window=50).mean().iloc[-1]) - 1) * 100,
                'bb_lower_dev': calculate_bb_deviation(hist['Close']),
                'decline_from_peak': decline_from_peak,
                'rsi': calculate_rsi(hist['Close']).iloc[-1],
                'macd_cross': macd_crossover(macd, signal_line),
                'macd_latest': macd.iloc[-1], 'signal_latest': signal_line.iloc[-1],
                'volume_ratio': calculate_volume_ratio(hist['Volume']),
                'close_prices': hist['Close']
            }
    return stock_data

# --- 분석/해석 함수 ---
def calculate_mhi_score(data):
    scores = {}
    price_flow_ratio = (data.get('sp500_close', 0) / data.get('sp500_ma20', 1)) - 1
    scores['price_flow'] = (price_flow_ratio + 0.03) / 0.06 * 100 if -0.03 <= price_flow_ratio <= 0.03 else (100 if price_flow_ratio > 0.03 else 0)
    rsi = data.get('sp500_rsi', 50)
    scores['trend'] = 100 - ((rsi - 30) / 40 * 100) if 30 <= rsi <= 70 else (100 if rsi < 30 else 0)
    vix_score = 100 - min(max((data.get('vix', 20) - 12) / 20 * 100, 0), 100)
    tnx_change = data.get('tnx_change', 0)
    tnx_score = (3 - tnx_change) / 6 * 100 if -3 <= tnx_change <= 3 else (100 if tnx_change < -3 else 0)
    scores['liquidity'] = (vix_score * 0.6) + (tnx_score * 0.4)
    scores['sentiment'] = 100 - data.get('fng_value', 50)
    weights = {'price_flow': 0.3, 'trend': 0.2, 'liquidity': 0.3, 'sentiment': 0.2}
    final_score = sum(scores.get(k, 50) * v for k, v in weights.items())
    return final_score, scores

# --- UI 렌더링 ---
st.title("🤖 ROgicX 작전 본부 v8.4")

# --- 모듈 1: 시장 종합 체감 지수 (MHI) ---
st.subheader("🌐 시장 종합 체감 지수 (MHI)")
market_data = get_market_health_data()
mhi_score, component_scores = calculate_mhi_score(market_data)

if mhi_score >= 80: status, color = "� 강세", "blue"
elif mhi_score >= 60: status, color = "🟢 양호", "green"
elif mhi_score >= 40: status, color = "🟡 중립", "orange"
elif mhi_score >= 20: status, color = "🟠 주의", "red"
else: status, color = "🔴 위험", "violet"

st.markdown(f"### 현재 MHI 점수: **:{color}[{mhi_score:.1f}점]** ({status})")

with st.expander("ℹ️ MHI 세부 지표 및 해석"):
    st.markdown("...") # 생략

st.divider()

# --- 포트폴리오 데이터를 로드하여 나머지 모듈 표시 ---
df = load_data_from_gsheet()

if df is not None:
    tickers_to_fetch = df['종목코드'].dropna().unique().tolist()
    stock_names_to_fetch = df['종목명'].dropna().unique().tolist()
    stock_data = get_stock_data(tickers_to_fetch, stock_names_to_fetch)

    # --- 모듈 2: 아군 현황판 (v8.4 로직 수정) ---
    st.subheader("📊 아군 현황판")
    
    cash_df = df[(df['자산티어'] == '현금') & (df['종목명'] == 'CMA')]
    available_cash = cash_df['현재평가금액'].sum()

    invest_categories = ['Tier 1', 'Tier 1.5', 'Tier 2', 'Tier 3']
    invested_df = df[df['자산티어'].isin(invest_categories)].copy()
    total_invested_value = invested_df['현재평가금액'].sum()

    tier_summary = invested_df.groupby('자산티어')['현재평가금액'].sum().reset_index()
    if total_invested_value > 0:
        tier_summary['현재 비중'] = (tier_summary['현재평가금액'] / total_invested_value) * 100
    else:
        tier_summary['현재 비중'] = 0

    # 목표 비중 하드코딩
    target_percentages = {'Tier 1': 40.0, 'Tier 1.5': 25.0, 'Tier 2': 25.0, 'Tier 3': 10.0}
    tier_summary['목표 비중'] = tier_summary['자산티어'].map(target_percentages).fillna(0)

    core_assets = ['Tier 1', 'Tier 1.5']
    core_current_percentage = tier_summary[tier_summary['자산티어'].isin(core_assets)]['현재 비중'].sum()
    core_target_percentage = tier_summary[tier_summary['자산티어'].isin(core_assets)]['목표 비중'].sum()
    core_gap = core_current_percentage - core_target_percentage
    
    action_proposal = ""
    if core_gap < -10:
        action_proposal = f"**코어 비중이 목표 대비 {abs(core_gap):.1f}% 부족합니다.**"
    elif total_invested_value > 0 and (available_cash / (total_invested_value + available_cash) > 0.3):
        action_proposal = "**가용 실탄이 충분합니다. 적극적인 기회 탐색이 필요합니다.**"
    else:
        action_proposal = "자산 배분이 안정적입니다."

    st.markdown("##### 포트폴리오 진단"); st.info(f"""- **자산 배분:** {action_proposal}\n- **가용 실탄:** **{available_cash:,.0f}원**의 작전 자금 준비 완료.""")
    
    tier_order = ['Tier 1', 'Tier 1.5', 'Tier 2', 'Tier 3']
    tier_summary['자산티어'] = pd.Categorical(tier_summary['자산티어'], categories=tier_order, ordered=True)
    tier_summary = tier_summary.sort_values('자산티어')
    
    fig = go.Figure()
    for index, row in tier_summary.iterrows():
        tier, current_val, target_val = row['자산티어'], row['현재 비중'], row['목표 비중']
        
        if current_val >= target_val:
            fig.add_trace(go.Bar(x=[tier], y=[current_val], name='현재 비중', marker_color='#1f77b4', text=f"{current_val:.1f}%", textposition='outside'))
            fig.add_trace(go.Bar(x=[tier], y=[target_val], name='목표 비중', marker_color='#ff7f0e', text=f"{target_val:.1f}%", textposition='inside'))
        else:
            fig.add_trace(go.Bar(x=[tier], y=[target_val], name='목표 비중', marker_color='#ff7f0e', text=f"{target_val:.1f}%", textposition='outside'))
            fig.add_trace(go.Bar(x=[tier], y=[current_val], name='현재 비중', marker_color='#1f77b4', text=f"{current_val:.1f}%", textposition='inside'))
            
    fig.update_layout(barmode="overlay", title="운용 자산 티어별 비중 비교", yaxis_title="비중 (%)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ==============================================================================
    # --- 모듈 3: 지능형 기회 포착 레이더 ---
    # ==============================================================================
    st.subheader("📡 지능형 기회 포착 레이더")

    sensitivity_level = st.radio(
        "감시 민감도 설정:",
        ('엄격하게 (Strict)', '중간 (Normal)', '널널하게 (Loose)'),
        index=1, horizontal=True, key='sensitivity'
    )
    
    sensitivity_map = {'엄격하게 (Strict)': 'Strict', '중간 (Normal)': 'Normal', '널널하게 (Loose)': 'Loose'}
    selected_sensitivity = sensitivity_map[sensitivity_level]

    sensitivity_params = {
        'Strict': {'tier1_bb_dev': -3, 'tier1_ma_dev': -10, 'tier1_peak_decline': -15, 'tier1_rsi': 35, 'tier1_vol': 1.5, 'tier1_5_bb_dev': -5, 'tier1_5_ma_dev': -15, 'tier1_5_peak_decline': -20, 'tier1_5_rsi': 32, 'tier1_5_vol': 1.8, 'tier2_bb_dev': -6, 'tier2_ma_dev': -18, 'tier2_peak_decline': -25, 'tier2_rsi': 30, 'tier2_vol': 2.0, 'tier4_bb_dev': -4, 'tier4_ma_dev': -12, 'tier4_peak_decline': -18, 'tier4_rsi': 30, 'tier4_vol': 1.5},
        'Normal': {'tier1_bb_dev': -2, 'tier1_ma_dev': -6, 'tier1_peak_decline': -10, 'tier1_rsi': 40, 'tier1_vol': 1.2, 'tier1_5_bb_dev': -4, 'tier1_5_ma_dev': -12, 'tier1_5_peak_decline': -18, 'tier1_5_rsi': 38, 'tier1_5_vol': 1.3, 'tier2_bb_dev': -5, 'tier2_ma_dev': -15, 'tier2_peak_decline': -20, 'tier2_rsi': 35, 'tier2_vol': 1.5, 'tier4_bb_dev': -3, 'tier4_ma_dev': -10, 'tier4_peak_decline': -15, 'tier4_rsi': 35, 'tier4_vol': 1.2},
        'Loose':  {'tier1_bb_dev': -1, 'tier1_ma_dev': -5, 'tier1_peak_decline': -8, 'tier1_rsi': 45, 'tier1_vol': 1.0, 'tier1_5_bb_dev': -3, 'tier1_5_ma_dev': -10, 'tier1_5_peak_decline': -15, 'tier1_5_rsi': 42, 'tier1_5_vol': 1.1, 'tier2_bb_dev': -4, 'tier2_ma_dev': -12, 'tier2_peak_decline': -18, 'tier2_rsi': 40, 'tier2_vol': 1.2, 'tier4_bb_dev': -2, 'tier4_ma_dev': -8, 'tier4_peak_decline': -12, 'tier4_rsi': 40, 'tier4_vol': 1.0}
    }
    current_params = sensitivity_params[selected_sensitivity]

    with st.expander("ℹ️ 레이더 규칙: '3중 하이브리드' 시스템"):
        st.markdown(f"""
        **'지능형 레이더 v8.4'** 은 **3중 하이브리드 가격 매력도**를 통해 분석의 정확성을 극대화합니다.

        ---
        #### ✅ 4대 분석 체크리스트
        - **가격 매력도:** `BB 하단 이탈` OR `MA 이탈` OR `3개월 고점 대비 하락` 중 하나라도 충족하면 통과.
        - **에너지 응축:** RSI 지표 (과매도 상태에 진입했는가?)
        - **추세 전환:** MACD 골든크로스 (하락을 멈추고 상승으로 전환하는가?)
        - **시장 동의:** 거래량 비율 (시장의 관심이 쏠려있는가?)
        ---
        #### 🚦 상태 우선순위
        `🟢 포착` > `🟡 감시` > `⚡ 변동성` > `⚪️ 안정` > `⚠️ 과열`
        ---
        #### 🎯 현재 민감도 기준 ('{sensitivity_level}')
        - **Tier 1:** `가격 매력도` (BB ≤ {current_params['tier1_bb_dev']}% or MA ≤ {current_params['tier1_ma_dev']}% or 고점 ≤ {current_params['tier1_peak_decline']}%) + `에너지 응축` (RSI ≤ {current_params['tier1_rsi']})
        - **Tier 1.5:** `가격 매력도` (BB ≤ {current_params['tier1_5_bb_dev']}% or MA ≤ {current_params['tier1_5_ma_dev']}% or 고점 ≤ {current_params['tier1_5_peak_decline']}%) + `에너지 응축` (RSI ≤ {current_params['tier1_5_rsi']}) + **`추세 전환`**
        - **Tier 2:** `가격 매력도` (BB ≤ {current_params['tier2_bb_dev']}% or MA ≤ {current_params['tier2_ma_dev']}% or 고점 ≤ {current_params['tier2_peak_decline']}%) + `에너지 응축` (RSI ≤ {current_params['tier2_rsi']}) + **`추세 전환`**
        - **포착(🟢) 조건:** 위 감시(🟡) 조건 충족 + `시장 동의` (거래량)
        """)

    radar_list = []
    monitor_df = df[~df['자산티어'].isin(['현금', '기반'])].copy()

    for index, row in monitor_df.iterrows():
        ticker, tier = row['종목코드'], row['자산티어']
        if tier not in ['Tier 1', 'Tier 1.5', 'Tier 2', 'Tier 4']: tier = 'Tier 4'

        stock_info = stock_data.get(ticker)
        if stock_info:
            analysis_result = analyze_stock_v8_4(stock_info, tier, current_params)
            if analysis_result: radar_list.append(analysis_result)

    if radar_list:
        radar_df = pd.DataFrame(radar_list)
        radar_df_display = radar_df[['상태', '종목명', '티어', '3일 변동률', '가격 매력도', '에너지 응축', '추세 전환', '시장 동의', 'status_order']]
        radar_df_display = radar_df_display.sort_values(by='status_order').drop(columns=['status_order'])
        st.dataframe(radar_df_display, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ **레이더 데이터 없음:** 분석 가능한 종목 데이터가 없습니다.")
    
    st.divider()
    st.subheader("📋 포트폴리오 상세 내역")
    st.dataframe(df, hide_index=True)
    
    # --- 모듈 4: GEM: Finance 보고용 브리핑 생성 ---
    st.subheader("✨ GEM: Finance 보고용 브리핑 생성")
    if st.button("원클릭 브리핑 생성"):
        guidance = f"{status} ({mhi_score:.1f}점)"
        
        if 'radar_df' in locals() and not radar_df.empty:
            significant_alerts = radar_df[radar_df['상태'].isin(['🟢 포착', '🟡 감시', '⚡ 변동성'])]
            
            if not significant_alerts.empty:
                alerts_text = ""
                for _, row in significant_alerts.iterrows():
                    alerts_text += (f"- **{row['종목명']}** ({row['티어']}): {row['상태']} | "
                                    f"가격: {row['가격 매력도']}, "
                                    f"에너지: {row['에너지 응축']}, "
                                    f"추세: {row['추세 전환']}, "
                                    f"거래량: {row['시장 동의']}\n")
            else:
                alerts_text = "현재 포착된 유의미한 매수/감시/변동성 신호는 없습니다."
        else:
            alerts_text = "레이더 데이터가 없어 분석할 신호가 없습니다."

        briefing = f"""
### 1. 전장 상황 브리핑 (MHI)
- **시장 종합 체감 지수:** {guidance}

### 2. 기회 포착 레이더 현황
{alerts_text}

### 3. 질문
위 상황을 참고 및 검증하고, 오늘의 증시를 보고해주세요.
"""
        st.text_area("아래 내용을 복사하여 GEM: Finance에 질문하세요.", briefing, height=300)

else:
    st.info("컨트롤 패널에 포트폴리오 엑셀 파일을 업로드하면 아군 현황 및 기회 포착 레이더가 활성화됩니다.")