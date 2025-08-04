import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
import gspread
from gspread_dataframe import get_as_dataframe
import numpy as np # numpy import 추가

# --- 페이지 설정 ---
st.set_page_config(page_title="ROgicX 작전 본부 v6.0", page_icon="🤖", layout="wide")

# ==============================================================================
# --- 모든 계산 함수 (v6.0에 맞게 일부 수정) ---
# ==============================================================================
def calculate_rsi(close_prices, window=14):
    """RSI(상대강도지수) 계산"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    # loss가 0일 경우 RSI를 100으로 설정하여 '과매수' 상태로 해석
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(100)

def calculate_macd_signal(close_prices, fast=12, slow=26, signal=9):
    """최근 3일 내 MACD 골든크로스 발생 여부 확인"""
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # 최근 3일 동안의 MACD와 Signal Line을 확인
    recent_macd = macd.iloc[-3:]
    recent_signal = signal_line.iloc[-3:]
    
    # (MACD가 Signal 아래에 있다가 위로 올라오는) 골든크로스 패턴 확인
    crossed_up = (recent_macd.shift(1) < recent_signal.shift(1)) & (recent_macd > recent_signal)
    
    return crossed_up.any()

def calculate_volume_ratio(volume_series, window=20):
    """최근 거래량 / 20일 평균 거래량 비율 계산"""
    if len(volume_series) < window: return 1.0
    # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon)을 더함
    avg_volume = volume_series.rolling(window=window).mean().iloc[-1]
    last_volume = volume_series.iloc[-1]
    return (last_volume / avg_volume) if avg_volume > 1e-6 else 1.0

def calculate_bb_deviation(close_prices, window=20, num_std=2):
    """볼린저 밴드 이탈도 계산"""
    ma = close_prices.rolling(window=window).mean()
    std = close_prices.rolling(window=window).std()
    lower_band = ma - (num_std * std)
    upper_band = ma + (num_std * std)
    lower_dev = ((close_prices - lower_band) / lower_band * 100).iloc[-1] if lower_band.iloc[-1] != 0 else 0
    upper_dev = ((close_prices - upper_band) / upper_band * 100).iloc[-1] if upper_band.iloc[-1] != 0 else 0
    return lower_dev, upper_dev

def calculate_atr(high_prices, low_prices, close_prices, window=14):
    """ATR(Average True Range) 계산"""
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift(1))
    tr3 = abs(low_prices - close_prices.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.iloc[-1] if len(atr) > 0 else 0

def get_price_change_rate(close_prices, window=3):
    """N일간 가격 변동률 계산"""
    if len(close_prices) < window:
        return 0
    return (close_prices.iloc[-1] / close_prices.iloc[-window] - 1) * 100

# v6.0에서 더 이상 사용되지 않는 함수들: is_crash_detected, is_surge_detected, get_price_change_stage, is_buy_signal

# ==============================================================================
# --- v6.0 핵심 분석 모듈 (신규 추가) ---
# ==============================================================================
def analyze_stock_v6(stock_info, tier):
    """
    '지능형 레이더 v6.0'의 규칙에 따라 종목을 분석하고 상태를 진단합니다.
    """
    if not stock_info or 'close_prices' not in stock_info or stock_info['close_prices'].empty:
        return None

    # --- 1. 개별 규칙(체크리스트) 통과 여부 확인 ---
    rsi = stock_info.get('rsi', 50)
    valuation_pass = rsi <= 35
    
    macd_cross = stock_info.get('macd_cross', False)
    trend_reversal_pass = macd_cross
    
    volume_ratio = stock_info.get('volume_ratio', 0)
    volume_check_pass = volume_ratio >= 1.5

    # --- 2. 가격 변동성 분석 ---
    price_change_rate = get_price_change_rate(stock_info['close_prices'], window=3)
    
    # --- 3. 최종 상태(Status) 결정: 티어별 규칙 적용 ---
    status = "⚪️ 안정"
    status_order = 3
    
    is_captured = False
    if tier == 'Tier 1':
        if valuation_pass and volume_check_pass:
            is_captured = True
    elif tier == 'Tier 2':
        if valuation_pass and trend_reversal_pass and volume_check_pass:
            is_captured = True

    if is_captured:
        status, status_order = "🟢 포착", 1
    elif (tier == 'Tier 1' and valuation_pass) or \
         (tier == 'Tier 2' and valuation_pass and trend_reversal_pass):
        status, status_order = "🟡 감시", 2

    if price_change_rate >= 7:
        status, status_order = "⚠️ 과열", 4
    elif price_change_rate <= -7:
        status, status_order = "⚡ 변동성", 5

    # --- 4. 최종 결과 정리 ---
    return {
        '상태': status,
        '종목명': stock_info['name'],
        '티어': tier,
        '가격 변동': f"{price_change_rate:.1f}%",
        '가치 평가': f"{'✅' if valuation_pass else '❌'} (RSI: {rsi:.1f})",
        '추세 전환': f"{'✅' if trend_reversal_pass else '❌'}",
        '거래량 확인': f"{'✅' if volume_check_pass else '❌'} ({volume_ratio:.1f}배)",
        'status_order': status_order
    }

# --- 데이터 로딩 함수 (기존과 동일) ---
@st.cache_data(ttl=600)
def load_data_from_gsheet():
    """구글 시트에서 포트폴리오 데이터를 로드합니다."""
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
        st.warning("secrets.toml 설정과 구글 시트 공유 설정을 다시 확인해주세요.")
        return None
    
def get_macro_data():
    """파일 업로드 없이 거시 지표만 가져옵니다. (안정성 강화)"""
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
        if len(dxy_data) >= 2:
            macro_data['dxy_change'] = (dxy_data.iloc[-1] / dxy_data.iloc[-2] - 1) * 100
        else:
            macro_data['dxy_change'] = 0
    except Exception: 
        macro_data['dxy_change'] = 0
        
    try:
        oil_data = yf.Ticker("CL=F").history(period='5d')['Close']
        if len(oil_data) >= 2:
            macro_data['oil_change'] = (oil_data.iloc[-1] / oil_data.iloc[-2] - 1) * 100
        else:
            macro_data['oil_change'] = 0
    except Exception: 
        macro_data['oil_change'] = 0
        
    return macro_data

@st.cache_data
def get_stock_data(tickers, stock_names):
    """개별 종목 데이터만 가져옵니다."""
    stock_data = {}
    # Ticker와 종목명을 매핑하는 딕셔너리 생성
    ticker_to_name = dict(zip(tickers, stock_names))

    valid_tickers = [t for t in tickers if t and isinstance(t, str) and t != 'CASH']
    for ticker in valid_tickers:
        try:
            hist = yf.Ticker(ticker).history(period='1y')
            if not hist.empty and len(hist) > 50:
                bb_lower_dev, bb_upper_dev = calculate_bb_deviation(hist['Close'])
                stock_data[ticker] = {
                    'name': ticker_to_name.get(ticker, ticker), # 종목명 추가
                    'deviation': ((hist['Close'].iloc[-1] / hist['Close'].rolling(window=50).mean().iloc[-1]) - 1) * 100,
                    'rsi': calculate_rsi(hist['Close']).iloc[-1],
                    'macd_cross': calculate_macd_signal(hist['Close']),
                    'volume_ratio': calculate_volume_ratio(hist['Volume']),
                    'bb_lower_dev': bb_lower_dev,
                    'bb_upper_dev': bb_upper_dev,
                    'atr': calculate_atr(hist['High'], hist['Low'], hist['Close']),
                    'close_prices': hist['Close']
                }
            else: stock_data[ticker] = None
        except Exception: stock_data[ticker] = None
    return stock_data

# --- 분석/해석 함수 (기존과 동일) ---
def calculate_opportunity_score(macro_data):
    reasons = {}
    fng_val, vix_val, dxy_change, oil_change = macro_data.get('fng_value'), macro_data.get('vix'), macro_data.get('dxy_change'), macro_data.get('oil_change')
    reasons['fng'] = 2 if fng_val is not None and fng_val <= 25 else 0
    reasons['vix'] = 2 if vix_val is not None and vix_val >= 30 else 0
    reasons['dxy'] = 1 if dxy_change is not None and dxy_change >= 0.5 else 0
    reasons['oil'] = 1 if oil_change is not None and oil_change <= -3.0 else 0
    return sum(reasons.values()), reasons

# --- UI 렌더링 ---
st.title("🤖 ROgicX 작전 본부 v6.0")


# --- 모듈 1: 전장 상황판 (기존과 동일) ---
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

    # --- 모듈 2: 아군 현황판 (기존과 동일) ---
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
    # --- 모듈 3: 지능형 기회 포착 레이더 (v6.0으로 전면 교체) ---
    # ==============================================================================
    st.subheader("📡 지능형 기회 포착 레이더 v6.0")

    with st.expander("ℹ️ v6.0 레이더 규칙: '종합 검진표' 시스템"):
        st.markdown("""
        **'지능형 레이더 v6.0'**은 각 종목의 상태를 다각도로 진단하는 **'종합 검진표'**입니다.
        단순 결과가 아닌, **분석 과정을 투명하게 공개**하여 MASTER의 최종 판단을 돕습니다.

        ---
        
        #### 🩺 5단계 상태 진단 시스템
        | 상태 | 아이콘 | 의미 |
        | :--- | :--- | :--- |
        | **포착** | 🟢 | 모든 매수 조건 충족. 즉각적인 분석이 필요한 **최우선 타겟**. |
        | **감시** | 🟡 | 핵심 조건은 충족했으나, 최종 확인(거래량 등)이 필요한 상태. |
        | **안정** | ⚪️ | 특별한 기회나 위협이 없는 '조용한' 상태. |
        | **과열** | ⚠️ | 단기 급등으로 추격 매수의 위험이 높은 상태. (3일간 7% 이상 상승) |
        | **변동성** | ⚡ | 최근 주가 변동성이 매우 커져 주의가 필요한 상태. (3일간 7% 이상 하락) |

        ---

        #### ✅ 체크리스트 항목별 기준
        - **가치 평가:** `RSI <= 35` 인가? (가격이 저렴한가?)
        - **추세 전환:** 최근 3일 내 `MACD 골든크로스`가 발생했는가? (하락이 멈췄는가?)
        - **거래량 확인:** `최근 거래량 >= 20일 평균의 1.5배` 인가? (시장의 관심이 있는가?)

        ---
        
        #### 🎯 티어별 '포착(🟢)' 규칙
        - **Tier 1 (코어 자산):** `가치 평가 ✅` AND `거래량 확인 ✅`
        - **Tier 2 (위성 자산):** `가치 평가 ✅` AND `추세 전환 ✅` AND `거래량 확인 ✅`
        """)

    radar_list = []
    # '현금', '기반' 티어를 제외한 모든 자산을 모니터링
    monitor_df = df[~df['자산티어'].isin(['현금', '기반'])].copy()

    for index, row in monitor_df.iterrows():
        ticker = row['종목코드']
        tier = row['자산티어']
        
        stock_info = stock_data.get(ticker)
        if stock_info:
            analysis_result = analyze_stock_v6(stock_info, tier)
            if analysis_result:
                radar_list.append(analysis_result)

    if radar_list:
        radar_df = pd.DataFrame(radar_list)
        radar_df = radar_df.sort_values(by='status_order').drop(columns=['status_order'])
        
        st.dataframe(
            radar_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "상태": st.column_config.TextColumn(width="small"),
                "종목명": st.column_config.TextColumn(width="small"),
                "티어": st.column_config.TextColumn(width="small"),
            }
        )
    else:
        st.warning("⚠️ **레이더 데이터 없음:** 분석 가능한 종목 데이터가 없습니다. 엑셀 파일의 종목코드를 확인해주세요.")
    
    st.divider()
    st.subheader("📋 포트폴리오 상세 내역")
    st.dataframe(df, hide_index=True)
    
    # --- 모듈 4: GEM: Finance 보고용 브리핑 생성 (v6.0 데이터 구조에 맞게 수정) ---
    st.subheader("✨ GEM: Finance 보고용 브리핑 생성")
    if st.button("원클릭 브리핑 생성"):
        guidance = "🔥 역매수 작전 고려" if total_score >= 7 else "🟡 기회 감시 강화" if total_score >= 4 else "⚪️ 훈련의 날"
        
        if 'radar_df' in locals() and not radar_df.empty:
            # 레이더에서 유의미한 신호('포착', '감시')만 필터링
            significant_alerts = radar_df[radar_df['상태'].isin(['🟢 포착', '🟡 감시'])]
            
            if not significant_alerts.empty:
                alerts_text = ""
                for _, row in significant_alerts.iterrows():
                    # v6.0의 체크리스트 결과를 브리핑에 포함
                    alerts_text += (f"- **{row['종목명']}** ({row['티어']}): {row['상태']} | "
                                    f"가치: {row['가치 평가']} | "
                                    f"추세: {row['추세 전환']} | "
                                    f"거래량: {row['거래량 확인']}\n")
            else:
                alerts_text = "현재 포착된 유의미한 매수/감시 신호는 없습니다."
        else:
            alerts_text = "레이더 데이터가 없어 분석할 신호가 없습니다."

        briefing = f"""
### 1. 전장 상황 브리핑
- **종합 기회 지수:** {total_score}점
- **행동 지침:** {guidance}

### 2. 기회 포착 레이더 현황 (v6.0)
{alerts_text}

### 3. 질문
위 상황을 참고 및 검증하고, 오늘의 증시를 보고해주세요.
"""
        st.text_area("아래 내용을 복사하여 GEM: Finance에 질문하세요.", briefing, height=300)

else:
    st.info("컨트롤 패널에 포트폴리오 엑셀 파일을 업로드하면 아군 현황 및 기회 포착 레이더가 활성화됩니다.")
