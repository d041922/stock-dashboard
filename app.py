import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
import gspread
from gspread_dataframe import get_as_dataframe
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="ROgicX 작전 본부 v6.9", page_icon="🤖", layout="wide")

# ==============================================================================
# --- 모든 계산 함수 ---
# ==============================================================================
def calculate_rsi(close_prices, window=14):
    """RSI(상대강도지수) 계산"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(100)

def calculate_volume_ratio(volume_series, window=20):
    """최근 거래량 / 20일 평균 거래량 비율 계산"""
    if len(volume_series) < window: return 1.0
    avg_volume = volume_series.rolling(window=window).mean().iloc[-1]
    last_volume = volume_series.iloc[-1]
    return (last_volume / avg_volume) if avg_volume > 1e-6 else 1.0

def get_price_change_rate(close_prices, window=3):
    """N일간 가격 변동률 계산"""
    if len(close_prices) < window:
        return 0
    return (close_prices.iloc[-1] / close_prices.iloc[-window] - 1) * 100

def calculate_bb_deviation(close_prices, window=20, num_std=2):
    """볼린저 밴드 하단선 대비 현재가의 이격도를 계산합니다."""
    ma = close_prices.rolling(window=window).mean()
    std = close_prices.rolling(window=window).std()
    lower_band = ma - (num_std * std)
    last_price = close_prices.iloc[-1]
    last_lower_band = lower_band.iloc[-1]
    if last_lower_band == 0: return 0.0
    deviation = ((last_price / last_lower_band) - 1) * 100
    return deviation

# ==============================================================================
# --- v6.9 핵심 분석 모듈 (최종 버전) ---
# ==============================================================================
def analyze_stock_v6_9(stock_info, tier, params):
    """
    '지능형 레이더 v6.9'의 규칙에 따라 종목을 분석하고 상태를 진단합니다.
    """
    if not stock_info or 'close_prices' not in stock_info or stock_info['close_prices'].empty:
        return None

    # --- 1. 5대 분석 지표 추출 ---
    deviation = stock_info.get('deviation', 0)
    bb_lower_dev = stock_info.get('bb_lower_dev', 0)
    rsi = stock_info.get('rsi', 50)
    macd_cross = stock_info.get('macd_cross', False)
    macd_latest = stock_info.get('macd_latest', 0)
    signal_latest = stock_info.get('signal_latest', 0)
    volume_ratio = stock_info.get('volume_ratio', 0)
    price_change_rate = get_price_change_rate(stock_info['close_prices'], window=3)

    # --- 2. 티어별 규칙 적용 ---
    tier_num_str = tier[5] if len(tier) > 5 and tier.startswith('Tier') else '4'
    
    price_attractive_bb = (bb_lower_dev <= params[f'tier{tier_num_str}_bb_dev'])
    price_attractive_ma = (deviation <= params[f'tier{tier_num_str}_ma_dev'])
    price_attractive = price_attractive_bb or price_attractive_ma

    energy_condensed = (rsi <= params[f'tier{tier_num_str}_rsi'])
    market_agreed = (volume_ratio >= params[f'tier{tier_num_str}_vol'])

    is_watching = False
    is_captured = False
    if tier in ['Tier 1', 'Tier 4']:
        is_watching = price_attractive and energy_condensed
        if is_watching and market_agreed:
            is_captured = True
    elif tier == 'Tier 2':
        is_watching = price_attractive and energy_condensed and macd_cross
        if is_watching and market_agreed:
            is_captured = True

    # --- 3. 상태 설명 및 수치 텍스트 생성 ---
    if price_attractive:
        price_desc_parts = []
        if price_attractive_bb: price_desc_parts.append(f"BB({bb_lower_dev:.1f}%)")
        if price_attractive_ma: price_desc_parts.append(f"MA({deviation:.1f}%)")
        price_text = " ".join(price_desc_parts)
    else:
        price_text = f"기준 미달 (BB:{bb_lower_dev:.1f}%, MA:{deviation:.1f}%)"
    
    energy_desc = "과매도" if rsi <= 35 else "과열" if rsi >= 65 else "중립"
    energy_text = f"{energy_desc} (RSI:{rsi:.1f})"

    if macd_cross:
        trend_text = "상승 전환"
    else:
        trend_text = "상승 추세" if macd_latest > signal_latest else "하락 추세"

    volume_desc = "급증" if volume_ratio >= 1.5 else "부족" if volume_ratio < 1.0 else "평균"
    volume_text = f"{volume_desc} ({volume_ratio:.1f}배)"

    # --- 4. 최종 상태 및 우선순위 결정 ---
    status, status_order = "⚪️ 안정", 4
    if is_watching: status, status_order = "🟡 감시", 2
    if is_captured: status, status_order = "🟢 포착", 1

    if price_change_rate <= -7 and status == "⚪️ 안정": status, status_order = "⚡ 변동성", 3
    if price_change_rate >= 7: status, status_order = "⚠️ 과열", 5

    # --- 5. 최종 결과 포맷팅 ---
    return {
        '상태': status,
        '종목명': stock_info['name'],
        '티어': tier,
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
        df = get_as_dataframe(worksheet, evaluate_formulas=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        return df
    except Exception as e:
        st.error(f"Google Sheets 데이터를 불러오는 데 실패했습니다: {e}")
        return None

def get_macro_data():
    macro_data = {}
    try:
        fng_response = requests.get("https://api.alternative.me/fng/?limit=1")
        macro_data['fng_value'] = int(fng_response.json()['data'][0]['value'])
    except Exception: macro_data['fng_value'] = None
    try:
        macro_data['vix'] = yf.Ticker("^VIX").history(period='1d')['Close'][0]
    except Exception: macro_data['vix'] = None
    try:
        dxy_data = yf.Ticker("DX-Y.NYB").history(period='5d')['Close']
        if len(dxy_data) >= 2: macro_data['dxy_change'] = (dxy_data.iloc[-1] / dxy_data.iloc[-2] - 1) * 100
        else: macro_data['dxy_change'] = 0
    except Exception: macro_data['dxy_change'] = 0
    try:
        oil_data = yf.Ticker("CL=F").history(period='5d')['Close']
        if len(oil_data) >= 2: macro_data['oil_change'] = (oil_data.iloc[-1] / oil_data.iloc[-2] - 1) * 100
        else: macro_data['oil_change'] = 0
    except Exception: macro_data['oil_change'] = 0
    return macro_data

@st.cache_data
def get_stock_data(tickers, stock_names):
    stock_data = {}
    ticker_to_name = dict(zip(tickers, stock_names))
    valid_tickers = [t for t in tickers if t and isinstance(t, str) and t != 'CASH']
    for ticker in valid_tickers:
        try:
            hist = yf.Ticker(ticker).history(period='1y')
            if not hist.empty and len(hist) > 50:
                # MACD Calculation for detailed trend status
                exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
                exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=9, adjust=False).mean()
                
                recent_macd = macd.iloc[-3:]
                recent_signal = signal_line.iloc[-3:]
                crossed_up = (recent_macd.shift(1) < recent_signal.shift(1)) & (recent_macd > recent_signal)

                stock_data[ticker] = {
                    'name': ticker_to_name.get(ticker, ticker),
                    'deviation': ((hist['Close'].iloc[-1] / hist['Close'].rolling(window=50).mean().iloc[-1]) - 1) * 100,
                    'bb_lower_dev': calculate_bb_deviation(hist['Close']),
                    'rsi': calculate_rsi(hist['Close']).iloc[-1],
                    'macd_cross': crossed_up.any(),
                    'macd_latest': macd.iloc[-1],
                    'signal_latest': signal_line.iloc[-1],
                    'volume_ratio': calculate_volume_ratio(hist['Volume']),
                    'close_prices': hist['Close']
                }
            else: stock_data[ticker] = None
        except Exception as e:
            st.error(f"Failed to get data for {ticker}: {e}")
            stock_data[ticker] = None
    return stock_data

# --- 분석/해석 함수 ---
def calculate_opportunity_score(macro_data):
    reasons = {}
    fng_val, vix_val, dxy_change, oil_change = macro_data.get('fng_value'), macro_data.get('vix'), macro_data.get('dxy_change'), macro_data.get('oil_change')
    reasons['fng'] = 2 if fng_val is not None and fng_val <= 25 else 0
    reasons['vix'] = 2 if vix_val is not None and vix_val >= 30 else 0
    reasons['dxy'] = 1 if dxy_change is not None and dxy_change >= 0.5 else 0
    reasons['oil'] = 1 if oil_change is not None and oil_change <= -3.0 else 0
    return sum(reasons.values()), reasons

# --- UI 렌더링 ---
st.title("🤖 ROgicX 작전 본부 v6.9 (Final)")

# --- 모듈 1: 전장 상황판 ---
st.subheader("🌐 전장 상황판")
macro_data = get_macro_data()
total_score, score_reasons = calculate_opportunity_score(macro_data)
with st.expander("ℹ️ 전장 상황판 지표 해석"):
    st.markdown("""
    - **공포&탐욕 지수:** 시장의 심리를 나타냅니다. '극심한 공포'는 역발상 투자의 기회가 될 수 있습니다. **(≤25시 +2점)**
    - **VIX:** 시장의 변동성(불안감)을 나타냅니다. 지수가 높을수록 시장이 불안하다는 뜻이며, 이는 종종 좋은 자산을 싸게 살 기회를 의미합니다. **(≥30시 +2점)**
    - **달러인덱스:** 달러의 가치. 급등은 안전자산 선호 심리를 의미하며, 시장의 불안감을 나타내는 신호입니다. **(전일비 ≥+0.5%시 +1점)**
    - **WTI유가:** 국제 유가. 급락은 경기 침체에 대한 우려를 반영하며, 주식 시장의 하락 압력으로 작용할 수 있습니다. **(전일비 ≤-3.0%시 +1점)**
    - **종합 기회 지수:** 위 지표들을 종합하여, '역매수 작전'에 얼마나 유리한 환경인지를 점수화한 ROgicX 자체 지표입니다.
    """)
cols = st.columns(5)
fng_val, vix_val, dxy_change, oil_change = macro_data.get('fng_value'), macro_data.get('vix'), macro_data.get('dxy_change'), macro_data.get('oil_change')
with cols[0]:
    st.metric("공포&탐욕", f"{fng_val}" if fng_val else "N/A"); score_text = f"**점수: +{score_reasons['fng']}**"; st.markdown(f"{'🟢' if score_reasons['fng']>0 else '⚪️'} {score_text}")
with cols[1]:
    st.metric("VIX", f"{vix_val:.2f}" if vix_val else "N/A"); score_text = f"**점수: +{score_reasons['vix']}**"; st.markdown(f"{'🟢' if score_reasons['vix']>0 else '⚪️'} {score_text}")
with cols[2]:
    st.metric("달러인덱스(%)", f"{dxy_change:+.2f}%"); score_text = f"**점수: +{score_reasons['dxy']}**"; st.markdown(f"{'🟡' if score_reasons['dxy']>0 else '⚪️'} {score_text}")
with cols[3]:
    st.metric("WTI유가(%)", f"{oil_change:+.2f}%"); score_text = f"**점수: +{score_reasons['oil']}**"; st.markdown(f"{'🟡' if score_reasons['oil']>0 else '⚪️'} {score_text}")
with cols[4]:
    guidance = "🔥 역매수 작전 고려" if total_score >= 7 else "🟡 기회 감시 강화" if total_score >= 4 else "⚪️ 훈련의 날"
    st.metric("종합 기회 지수", f"**{total_score}**"); st.markdown(f"**{guidance}**")


st.divider()

# --- 포트폴리오 데이터를 로드하여 나머지 모듈 표시 ---
df = load_data_from_gsheet()

if df is not None:
    tickers_to_fetch = df['종목코드'].dropna().unique().tolist()
    stock_names_to_fetch = df['종목명'].dropna().unique().tolist()
    stock_data = get_stock_data(tickers_to_fetch, stock_names_to_fetch)
    total_score, _ = calculate_opportunity_score(macro_data)

    # --- 모듈 2: 아군 현황판 ---
    st.subheader("📊 아군 현황판")
    cash_df = df[(df['자산티어'] == '현금') & (df['종목명'] == 'CMA')]; available_cash = cash_df['현재평가금액'].sum()
    invested_df = df[~df['자산티어'].isin(['현금', '관심종목', 'Tier 4', '기반'])]; total_invested_value = invested_df['현재평가금액'].sum()
    tier_summary = invested_df.groupby('자산티어')['현재평가금액'].sum().reset_index(); tier_summary['현재비중(%)'] = (tier_summary['현재평가금액'] / total_invested_value) * 100 if total_invested_value > 0 else 0
    def parse_target(target_str):
        if not isinstance(target_str, str) or target_str.strip() in ['-', '']: return 0
        cleaned_str = target_str.replace('<', '').strip();
        if '-' in cleaned_str:
            try: parts = [float(p.strip()) for p in cleaned_str.split('-')]; return sum(parts) / len(parts) if len(parts) == 2 else 0
            except ValueError: return 0
        try: return float(cleaned_str)
        except ValueError: return 0
    target_df = df[['자산티어', '목표비중(%)']].dropna().drop_duplicates('자산티어'); target_df['목표비중(%)'] = target_df['목표비중(%)'].apply(parse_target)
    tier_summary = pd.merge(tier_summary, target_df, on='자산티어', how='left')
    core_gap = tier_summary[tier_summary['자산티어']=='Tier 1']['현재비중(%)'].iloc[0] - tier_summary[tier_summary['자산티어']=='Tier 1']['목표비중(%)'].iloc[0] if not tier_summary[tier_summary['자산티어']=='Tier 1'].empty else 0
    st.markdown("##### 종합 진단"); st.info(f"""- **자산 배분:** {'코어 비중 안정적.' if core_gap > -10 else f'**코어 비중이 목표 대비 {abs(core_gap):.1f}% 부족.**'}\n- **가용 실탄:** **{available_cash:,.0f}원**의 작전 자금 준비 완료.\n- **시장 상황:** 현재 기회 지수는 **{total_score}점**으로, **'{guidance.split('.')[0]}'** 입니다.""")
    tier_order = ['Tier 1', 'Tier 2', 'Tier 3']; tier_summary['자산티어'] = pd.Categorical(tier_summary['자산티어'], categories=tier_order, ordered=True); tier_summary = tier_summary.sort_values('자산티어')
    fig = go.Figure();
    for index, row in tier_summary.iterrows():
        tier, current_val, target_val = row['자산티어'], row['현재비중(%)'], row['목표비중(%)']
        show_legend_current, show_legend_target = (index == 0), (index == 0)
        if current_val >= target_val:
            fig.add_trace(go.Bar(x=[tier], y=[current_val], name='현재 비중', marker_color='#1f77b4', showlegend=show_legend_current, text=f"{current_val:.1f}%", textposition='outside'))
            fig.add_trace(go.Bar(x=[tier], y=[target_val], name='목표 비중', marker_color='lightgray', showlegend=show_legend_target, text=f"{target_val:.1f}%", textposition='inside'))
        else:
            fig.add_trace(go.Bar(x=[tier], y=[target_val], name='목표 비중', marker_color='lightgray', showlegend=show_legend_target, text=f"{target_val:.1f}%", textposition='outside'))
            fig.add_trace(go.Bar(x=[tier], y=[current_val], name='현재 비중', marker_color='#1f77b4', showlegend=show_legend_current, text=f"{current_val:.1f}%", textposition='inside'))
    fig.update_layout(title_text="운용 자산 티어별 비중 (기반 자산 제외)", barmode='overlay', yaxis_title='비중 (%)', legend_title_text=None, uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ==============================================================================
    # --- 모듈 3: 지능형 기회 포착 레이더 (v6.9) ---
    # ==============================================================================
    st.subheader("📡 지능형 기회 포착 레이더 v6.9")

    sensitivity_level = st.radio(
        "감시 민감도 설정:",
        ('엄격하게 (Strict)', '중간 (Normal)', '널널하게 (Loose)'),
        index=1, horizontal=True, key='sensitivity'
    )
    
    sensitivity_map = {
        '엄격하게 (Strict)': 'Strict', '중간 (Normal)': 'Normal', '널널하게 (Loose)': 'Loose'
    }
    selected_sensitivity = sensitivity_map[sensitivity_level]

    # 민감도 파라미터 재설계 (아마존 케이스 반영)
    sensitivity_params = {
        'Strict': {'tier1_bb_dev': -3, 'tier1_ma_dev': -10, 'tier1_rsi': 35, 'tier1_vol': 1.5, 'tier2_bb_dev': -6, 'tier2_ma_dev': -18, 'tier2_rsi': 30, 'tier2_vol': 2.0, 'tier4_bb_dev': -4, 'tier4_ma_dev': -12, 'tier4_rsi': 30, 'tier4_vol': 1.5},
        'Normal': {'tier1_bb_dev': -2, 'tier1_ma_dev': -6,  'tier1_rsi': 40, 'tier1_vol': 1.2, 'tier2_bb_dev': -5, 'tier2_ma_dev': -15, 'tier2_rsi': 35, 'tier2_vol': 1.5, 'tier4_bb_dev': -3, 'tier4_ma_dev': -10, 'tier4_rsi': 35, 'tier4_vol': 1.2},
        'Loose':  {'tier1_bb_dev': -1, 'tier1_ma_dev': -5,  'tier1_rsi': 45, 'tier1_vol': 1.0, 'tier2_bb_dev': -4, 'tier2_ma_dev': -12, 'tier2_rsi': 40, 'tier2_vol': 1.2, 'tier4_bb_dev': -2, 'tier4_ma_dev': -8,  'tier4_rsi': 40, 'tier4_vol': 1.0}
    }
    current_params = sensitivity_params[selected_sensitivity]

    with st.expander("ℹ️ v6.9 레이더 규칙: '하이브리드' 시스템"):
        st.markdown(f"""
        **'지능형 레이더 v6.9'** 은 **하이브리드 가격 매력도**와 **상세 수치**를 통해 분석의 정확성과 직관성을 극대화합니다.

        ---
        #### ✅ 4대 분석 체크리스트
        - **가격 매력도:** **볼린저 밴드** 또는 **50일 이평선** 기준 중 하나라도 충족하면 통과 (OR 조건)
        - **에너지 응축:** RSI 지표 (과매도 상태에 진입했는가?)
        - **추세 전환:** MACD 골든크로스 (하락을 멈추고 상승으로 전환하는가?)
        - **시장 동의:** 거래량 비율 (시장의 관심이 쏠려있는가?)
        ---
        #### 🚦 상태 우선순위
        `🟢 포착` > `🟡 감시` > `⚡ 변동성` > `⚪️ 안정` > `⚠️ 과열`
        - **변동성:** 다른 조건은 만족하지 못했으나, 3일간 -7% 이상 급락하여 주목이 필요한 상태.
        - **과열:** 다른 모든 조건보다 우선하는 리스크 관리 신호. (3일간 +7% 이상 급등)
        ---
        #### 🎯 현재 민감도 기준 ('{sensitivity_level}')
        - **Tier 1:**
            - `가격 매력도`: BB ≤ {current_params['tier1_bb_dev']}% 또는 MA ≤ {current_params['tier1_ma_dev']}%
            - `에너지 응축`: RSI ≤ {current_params['tier1_rsi']}
            - `시장 동의`: 거래량 ≥ {current_params['tier1_vol']}배
        - **Tier 2:**
            - `가격 매력도`: BB ≤ {current_params['tier2_bb_dev']}% 또는 MA ≤ {current_params['tier2_ma_dev']}%
            - `에너지 응축`: RSI ≤ {current_params['tier2_rsi']}
            - `추세 전환`: MACD 골든크로스 발생
            - `시장 동의`: 거래량 ≥ {current_params['tier2_vol']}배
        - **Tier 4:**
            - `가격 매력도`: BB ≤ {current_params['tier4_bb_dev']}% 또는 MA ≤ {current_params['tier4_ma_dev']}%
            - `에너지 응축`: RSI ≤ {current_params['tier4_rsi']}
            - `시장 동의`: 거래량 ≥ {current_params['tier4_vol']}배
        """)

    radar_list = []
    monitor_df = df[~df['자산티어'].isin(['현금', '기반'])].copy()

    for index, row in monitor_df.iterrows():
        ticker, tier = row['종목코드'], row['자산티어']
        if tier not in ['Tier 1', 'Tier 2', 'Tier 4']:
            tier = 'Tier 4'

        stock_info = stock_data.get(ticker)
        if stock_info:
            analysis_result = analyze_stock_v6_9(stock_info, tier, current_params)
            if analysis_result:
                radar_list.append(analysis_result)

    if radar_list:
        radar_df = pd.DataFrame(radar_list)
        radar_df_display = radar_df[['상태', '종목명', '티어', '가격 매력도', '에너지 응축', '추세 전환', '시장 동의', 'status_order']]
        radar_df_display = radar_df_display.sort_values(by='status_order').drop(columns=['status_order'])
        
        st.dataframe(
            radar_df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "상태": st.column_config.TextColumn("상태", width="small"),
                "종목명": st.column_config.TextColumn("종목명", width="small"),
                "티어": st.column_config.TextColumn("티어", width="small"),
            }
        )
    else:
        st.warning("⚠️ **레이더 데이터 없음:** 분석 가능한 종목 데이터가 없습니다.")
    
    st.divider()
    st.subheader("📋 포트폴리오 상세 내역")
    st.dataframe(df, hide_index=True)
    
    # --- 모듈 4: GEM: Finance 보고용 브리핑 생성 ---
    st.subheader("✨ GEM: Finance 보고용 브리핑 생성")
    if st.button("원클릭 브리핑 생성"):
        guidance = "🔥 역매수 작전 고려" if total_score >= 7 else "🟡 기회 감시 강화" if total_score >= 4 else "⚪️ 훈련의 날"
        
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
### 1. 전장 상황 브리핑
- **종합 기회 지수:** {total_score}점
- **행동 지침:** {guidance}

### 2. 기회 포착 레이더 현황 (v6.9)
{alerts_text}

### 3. 질문
위 상황을 참고 및 검증하고, 오늘의 증시를 보고해주세요.
"""
        st.text_area("아래 내용을 복사하여 GEM: Finance에 질문하세요.", briefing, height=300)

else:
    st.info("컨트롤 패널에 포트폴리오 엑셀 파일을 업로드하면 아군 현황 및 기회 포착 레이더가 활성화됩니다.")
