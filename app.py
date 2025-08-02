import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests

# --- 페이지 설정 ---
st.set_page_config(page_title="ROgicX 작전 본부 v3.5", page_icon="🤖", layout="wide")

# --- 기술 지표 계산 함수 ---
def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd_signal(close_prices, fast=12, slow=26, signal=9):
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    # 최근 3일 내 골든 크로스 발생 여부 체크
    return (macd.iloc[-3] < signal_line.iloc[-3] and macd.iloc[-1] > signal_line.iloc[-1]) or \
           (macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1])

def calculate_volume_ratio(volume_series, window=20):
    if len(volume_series) < window: return 1.0
    avg_volume = volume_series.rolling(window=window).mean().iloc[-1]
    last_volume = volume_series.iloc[-1]
    return (last_volume / avg_volume) if avg_volume > 0 else 1.0

def calculate_bb_deviation(close_prices, window=20, num_std=2):
    ma = close_prices.rolling(window=window).mean()
    std = close_prices.rolling(window=window).std()
    lower_band = ma - (num_std * std)
    upper_band = ma + (num_std * std)
    lower_dev = ((close_prices - lower_band) / lower_band * 100).iloc[-1] if lower_band.iloc[-1] != 0 else 0
    upper_dev = ((close_prices - upper_band) / upper_band * 100).iloc[-1] if upper_band.iloc[-1] != 0 else 0
    return lower_dev, upper_dev

def is_crash_detected(close_prices, window=3, threshold=-7):
    if len(close_prices) < window: return False
    recent_change = (close_prices.iloc[-1] / close_prices.iloc[-window] - 1) * 100
    return recent_change <= threshold

def is_surge_detected(close_prices, window=3, threshold=7):
    if len(close_prices) < window: return False
    recent_change = (close_prices.iloc[-1] / close_prices.iloc[-window] - 1) * 100
    return recent_change >= threshold

def calculate_atr(high_prices, low_prices, close_prices, window=14):
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift(1))
    tr3 = abs(low_prices - close_prices.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.iloc[-1] if len(atr) > 0 else 0

def get_price_change_rate(close_prices, window=3):
    """3일간 가격 변동률 계산"""
    if len(close_prices) < window:
        return 0
    return (close_prices.iloc[-1] / close_prices.iloc[-window] - 1) * 100

def get_price_change_stage(change_rate):
    """가격 변동률에 따른 단계 분류"""
    if change_rate <= -6:
        return "급락"
    elif -6 < change_rate <= -4:
        return "급락주의"
    elif -4 < change_rate < 4:
        return "안정"
    elif 4 <= change_rate < 6:
        return "급등주의"
    elif change_rate >= 6:
        return "급등"
    else:
        return "안정"

def is_buy_signal(stock_info, close_prices):
    """매수 신호 판단 (기존 지표 + 급락 단계 활용)"""
    deviation = stock_info['deviation']
    rsi = stock_info['rsi']
    macd_cross = stock_info['macd_cross']
    volume_ratio = stock_info['volume_ratio']
    bb_lower_dev = stock_info['bb_lower_dev']
    
    # 1) 기존 매수 조건
    rule1 = (deviation <= -8 and rsi <= 40)
    rule2 = (deviation <= -12)
    rule3 = (bb_lower_dev <= -5 and rsi <= 35)
    rule4 = (rsi <= 30)  # 과매도 구간 추가 조건
    rule5 = macd_cross  # 골든 크로스 여부
    
    base_buy_condition = (rule1 or rule2 or rule3 or rule4) and volume_ratio >= 1.2 and rule5
    
    # 2) 급락 단계 반영
    price_change_rate = get_price_change_rate(close_prices, window=3)
    price_stage = get_price_change_stage(price_change_rate)
    
    # 급락 또는 급락주의 구간이면 매수에 더욱 유리한 기회로 본다
    if price_stage in ["급락", "급락주의"]:
        # 진입 조건 완화 혹은 우위 신호 보강 가능
        base_buy_condition = base_buy_condition or (rsi <= 45 and volume_ratio >= 1.0)
    
    return base_buy_condition, price_stage



# --- 데이터 로딩 함수 ---
@st.cache_data(ttl=600)
def get_market_data(tickers):
    market_data = {}
    try:
        fng_response = requests.get("https://api.alternative.me/fng/?limit=1")
        market_data['fng_value'] = int(fng_response.json()['data'][0]['value'])
    except Exception: market_data['fng_value'] = None
    try: market_data['vix'] = yf.Ticker("^VIX").history(period='1d')['Close'][0]
    except Exception: market_data['vix'] = None
    try:
        dxy_data = yf.Ticker("DX-Y.NYB").history(period='2d')
        market_data['dxy_change'] = dxy_data['Close'].pct_change().iloc[-1] * 100
    except Exception: market_data['dxy_change'] = 0
    try:
        oil_data = yf.Ticker("CL=F").history(period='2d')
        market_data['oil_change'] = oil_data['Close'].pct_change().iloc[-1] * 100
    except Exception: market_data['oil_change'] = 0
    
    stock_data = {}
    valid_tickers = [t for t in tickers if t and isinstance(t, str) and t != 'CASH']
    for ticker in valid_tickers:
        try:
            hist = yf.Ticker(ticker).history(period='1y')
            if not hist.empty and len(hist) > 50:
                bb_lower_dev, bb_upper_dev = calculate_bb_deviation(hist['Close'])
                stock_data[ticker] = {
                    'deviation': ((hist['Close'].iloc[-1] / hist['Close'].rolling(window=50).mean().iloc[-1]) - 1) * 100,
                    'rsi': calculate_rsi(hist['Close']).iloc[-1],
                    'macd_cross': calculate_macd_signal(hist['Close']),
                    'volume_ratio': calculate_volume_ratio(hist['Volume']),
                    'bb_lower_dev': bb_lower_dev,
                    'bb_upper_dev': bb_upper_dev,
                    'atr': calculate_atr(hist['High'], hist['Low'], hist['Close']),
                    'is_crash': is_crash_detected(hist['Close']),
                    'is_surge': is_surge_detected(hist['Close']),
                    'close_prices': hist['Close']  # 전체 종가 시리즈 저장
                }
            else: stock_data[ticker] = None
        except Exception: stock_data[ticker] = None
    market_data['stocks'] = stock_data
    return market_data

# --- 분석/해석 함수 ---
def calculate_opportunity_score(market_data):
    reasons = {}
    fng_val, vix_val, dxy_change, oil_change = market_data.get('fng_value'), market_data.get('vix'), market_data.get('dxy_change'), market_data.get('oil_change')
    reasons['fng'] = 2 if fng_val is not None and fng_val <= 25 else 0
    reasons['vix'] = 2 if vix_val is not None and vix_val >= 30 else 0
    reasons['dxy'] = 1 if dxy_change is not None and dxy_change >= 0.5 else 0
    reasons['oil'] = 1 if oil_change is not None and oil_change <= -3.0 else 0
    return sum(reasons.values()), reasons

# --- UI 렌더링 ---
st.set_page_config(page_title="ROgicX 작전 본부 v3.5", page_icon="🤖", layout="wide")
st.title("🤖 ROgicX 작전 본부 v3.5 (Final)")
uploaded_file = st.file_uploader("포트폴리오 엑셀 파일을 업로드하세요.", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name=0)
    tickers_to_fetch = df['종목코드'].dropna().unique().tolist()
    market_data = get_market_data(tickers_to_fetch)
    total_score, score_reasons = calculate_opportunity_score(market_data)
    
    # (모듈 1, 2 UI 코드는 이전과 동일하여 생략)
    st.subheader("🌐 전장 상황판")
    cols = st.columns(5)
    fng_val, vix_val, dxy_change, oil_change = market_data.get('fng_value'), market_data.get('vix'), market_data.get('dxy_change'), market_data.get('oil_change')
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

    # --- 모듈 3: 지능형 기회 포착 레이더 (v4.1) ---
    st.subheader("📡 지능형 기회 포착 레이더 v4.1")
    with st.expander("ℹ️ 레이더 규칙 해석"):
        st.markdown("""
        **매수 신호:**
        - **🟢 포착:** 투자 철학에 부합하는 **강력한 매수 검토 신호**입니다. (규칙 충족 + 거래량 동반)
        - **🟡 주의:** 매수 관심권에 근접했으나, **거래량이 부족**하여 아직 시장의 동의를 얻지 못한 상태입니다.
        
        **급등락 단계:**
        - **⚡ 급락:** 3일간 -6% 이상 하락
        - **⚠️ 급락주의:** 3일간 -3~-6% 하락
        - **⚪️ 안정:** 3일간 -3~+3% 변동
        - **⚠️ 급등주의:** 3일간 +3~+6% 상승
        - **🚀 급등:** 3일간 +6% 이상 상승
        
        - **⚪️ 안정:** 특별 변동이 없거나, **규칙의 일부 조건만 충족**하여 아직 의미 있는 신호가 아닌 상태입니다.
        """)
    
    radar_list = []
    surge_crash_list = []
    # 기반 티어 제외하고 모니터링
    monitor_df = df[~df['자산티어'].isin(['현금', '기반'])].copy()
    
    for index, row in monitor_df.iterrows():
        ticker, tier, stock_info = row['종목코드'], row['자산티어'], market_data['stocks'].get(row['종목코드'])
        if not stock_info: 
            continue
        
        dev, rsi, macd_cross, vol_ratio = stock_info['deviation'], stock_info['rsi'], stock_info['macd_cross'], stock_info['volume_ratio']
        bb_lower_dev, bb_upper_dev = stock_info['bb_lower_dev'], stock_info['bb_upper_dev']
        close_prices = stock_info['close_prices']
        
        # 급등락 단계 계산
        price_change_rate = get_price_change_rate(close_prices, window=3)
        price_stage = get_price_change_stage(price_change_rate)
        
        # 급등락 상태 표시
        if price_stage == "급락":
            crash_surge_status = f"⚡ 급락({price_change_rate:.1f}%)"
        elif price_stage == "급락주의":
            crash_surge_status = f"⚠️ 급락주의({price_change_rate:.1f}%)"
        elif price_stage == "급등주의":
            crash_surge_status = f"⚠️ 급등주의({price_change_rate:.1f}%)"
        elif price_stage == "급등":
            crash_surge_status = f"🚀 급등({price_change_rate:.1f}%)"
        else:
            crash_surge_status = f"⚪️ 안정({price_change_rate:.1f}%)"
        
        # 매수 신호 판단 (개선된 로직)
        buy_signal, detected_stage = is_buy_signal(stock_info, close_prices)
        
        status, status_order, reason = "⚪️ 안정", 3, "기준 미달"
        
        if buy_signal:
            if vol_ratio >= 1.2:
                status, status_order, reason = "🟢 포착", 1, f"매수신호 + 거래량 {vol_ratio:.1f}배"
            else:
                status, status_order, reason = "🟡 주의", 2, f"매수신호 + 거래량 미달 ({vol_ratio:.1f}배)"
        else:
            # 실패 사유 구체화
            if dev > -8.0: 
                reason = f"이격도({dev:.1f}%) 안정"
            elif rsi > 40: 
                reason = f"RSI({rsi:.1f}) 안정"
            elif bb_lower_dev > -5.0: 
                reason = f"BB하단({bb_lower_dev:.1f}%) 안정"
            elif not macd_cross: 
                reason = "추세 전환 신호 없음"
            elif vol_ratio < 1.2: 
                reason = f"거래량 부족({vol_ratio:.1f}배)"
        
        radar_list.append({
            '상태': status, '종목명': row['종목명'], '티어': tier, 
            '급등락': crash_surge_status,
            '핵심 현황': f"이격도 {dev:.1f}%, RSI {rsi:.1f}, BB하단 {bb_lower_dev:.1f}%, BB상단 {bb_upper_dev:.1f}%", 
            '진단': reason, 'status_order': status_order
        })

    if radar_list:
        radar_df = pd.DataFrame(radar_list).sort_values(by='status_order').drop(columns=['status_order'])
        st.dataframe(radar_df, use_container_width=True, hide_index=True)
        
        # 급등락 신호 요약
        crash_stocks = [row['종목명'] for row in radar_list if "급락" in row['급등락']]
        surge_stocks = [row['종목명'] for row in radar_list if "급등" in row['급등락']]
        
        if crash_stocks:
            st.warning(f"⚠️ **급락 주의:** {', '.join(crash_stocks)}")
        if surge_stocks:
            st.info(f"📈 **급등 감지:** {', '.join(surge_stocks)}")
    else:
        st.warning("⚠️ **레이더 데이터 없음:** 분석 가능한 종목 데이터가 없습니다. 엑셀 파일의 종목코드가 올바른지 확인해주세요.")
    
    st.divider()
    st.subheader("📋 포트폴리오 상세 내역")
    st.dataframe(df, hide_index=True)