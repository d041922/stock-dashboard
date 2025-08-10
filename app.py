# ==============================================================================
# --- 1. 라이브러리 임포트 ---
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from gspread_dataframe import get_as_dataframe
from datetime import datetime
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

# ==============================================================================
# --- 2. 페이지 기본 설정 ---
# ==============================================================================
st.set_page_config(page_title="ROgicX 작전 본부 v17.0 (Real Data Integration)", page_icon="🎯", layout="wide")


# ==============================================================================
# --- 3. 기술적 지표 계산 함수 (공용) ---
# ==============================================================================
def calculate_rsi(close_prices, window=14):
    """RSI를 계산합니다."""
    if close_prices is None or len(close_prices) < window: return pd.Series([50] * len(close_prices))
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_macd(close_prices, span1=12, span2=26, signal_span=9):
    """MACD 라인과 시그널 라인을 계산합니다."""
    if close_prices is None or len(close_prices) < span2: return pd.Series(), pd.Series()
    ema12 = close_prices.ewm(span=span1, adjust=False).mean()
    ema26 = close_prices.ewm(span=span2, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(close_prices, window=20, num_std=2):
    """볼린저 밴드를 계산합니다."""
    if close_prices is None or len(close_prices) < window: return pd.Series(), pd.Series(), pd.Series()
    ma = close_prices.rolling(window=window).mean()
    std = close_prices.rolling(window=window).std()
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)
    return upper, ma, lower

# ==============================================================================
# --- 4. 매크로 리스크 분석 모듈 (실제 데이터 연동) ---
# ==============================================================================
@st.cache_data(ttl=600)
def fetch_fear_and_greed_index():
    """feargreedmeter.com에서 Fear & Greed Index를 스크레이핑합니다."""
    try:
        url = 'https://feargreedmeter.com/'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        fgi_element = soup.find('div', class_='text-center text-4xl font-semibold mb-1 text-white')
        if fgi_element and fgi_element.text.strip().isdigit():
            return int(fgi_element.text.strip())
        return 50 # 실패 시 중립 값 반환
    except Exception as e:
        st.warning(f"공포&탐욕 지수 로드 실패: {e}")
        return 50

@st.cache_data(ttl=600)
def fetch_put_call_ratio():
    """YCharts에서 Put/Call Ratio를 스크레이핑합니다."""
    try:
        url = 'https://ycharts.com/indicators/cboe_equity_put_call_ratio'
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        value_div = soup.find('div', class_='key-stat-val')
        if value_div:
            return float(value_div.text.strip())
        return 1.0 # 실패 시 중립 값 반환
    except Exception as e:
        st.warning(f"Put/Call 비율 로드 실패: {e}")
        return 1.0

@st.cache_data(ttl=600)
def get_macro_indicators():
    """주요 매크로 지표를 API 및 스크레이핑을 통해 가져옵니다."""
    tickers = {
        "VIX": "^VIX", "DXY": "DX-Y.NYB", "US10Y": "^TNX", "US30Y": "^TYX",
        "WTI": "CL=F", "Copper": "HG=F"
    }
    data = yf.download(list(tickers.values()), period="5d", progress=False)
    
    latest_data = {}
    for name, ticker in tickers.items():
        if not data['Close'][ticker].isnull().all():
            latest_data[name] = data['Close'][ticker].iloc[-1]
    
    if 'US10Y' in latest_data and 'US30Y' in latest_data:
        latest_data['Yield_Spread'] = latest_data['US30Y'] - latest_data['US10Y']
    
    # 실제 데이터로 대체
    latest_data['Fear_Greed'] = fetch_fear_and_greed_index()
    latest_data['Put_Call_Ratio'] = fetch_put_call_ratio()
    
    return latest_data

def calculate_risk_score(indicators, geo_risk):
    """각 지표를 점수화하고 종합 리스크 점수를 계산합니다."""
    scores = {}
    
    vix = indicators.get('VIX', 20)
    scores['VIX'] = min(max((vix - 15) / (40 - 15) * 100, 0), 100)
    
    fear_greed = indicators.get('Fear_Greed', 50)
    scores['Fear_Greed'] = 100 - min(max((fear_greed - 25) / (75 - 25) * 100, 0), 100)
    
    pcr = indicators.get('Put_Call_Ratio', 1.0)
    scores['Put_Call_Ratio'] = min(max((pcr - 0.8) / (1.2 - 0.8) * 100, 0), 100)
    
    stock_score = (scores['VIX'] * 0.5) + (scores['Fear_Greed'] * 0.3) + (scores['Put_Call_Ratio'] * 0.2)
    
    dxy = indicators.get('DXY', 104)
    scores['DXY'] = min(max((dxy - 102) / (108 - 102) * 100, 0), 100)
    
    tnx = indicators.get('US10Y', 4.0)
    scores['US10Y'] = min(max((tnx - 3.5) / (5.0 - 3.5) * 100, 0), 100)
    
    yield_spread = indicators.get('Yield_Spread', 0.1)
    scores['Yield_Spread'] = 100 - min(max((yield_spread - -0.5) / (0.5 - -0.5) * 100, 0), 100)
    
    bond_score = (scores['DXY'] * 0.4) + (scores['US10Y'] * 0.4) + (scores['Yield_Spread'] * 0.2)
    
    wti = indicators.get('WTI', 80)
    scores['WTI'] = min(max((wti - 70) / (100 - 70) * 100, 0), 100)
    
    copper = indicators.get('Copper', 4.5)
    scores['Copper'] = 100 - min(max((copper - 4.0) / (5.0 - 4.0) * 100, 0), 100)
    
    commodity_score = (scores['WTI'] * 0.6) + (scores['Copper'] * 0.4)

    geo_score = (geo_risk / 10) * 100
    scores['Geo_Risk'] = geo_score
    
    total_score = (stock_score * 0.4) + (bond_score * 0.3) + (commodity_score * 0.2) + (geo_score * 0.1)
    status = "위험" if total_score > 70 else "경계" if total_score > 40 else "안정"
    
    return total_score, status, scores

# ==============================================================================
# --- 5. 핵심 분석 및 점수 계산 모듈 (analyze_stock) ---
# ==============================================================================
def analyze_stock(stock_info, tier, params):
    """개별 종목의 데이터를 기반으로 상태와 점수를 분석하는 핵심 로직."""
    if not stock_info or stock_info.get('close_prices') is None: return None

    close_prices = stock_info['close_prices']
    if len(close_prices) < 3: return None
    price_change_rate = (close_prices.iloc[-1] / close_prices.iloc[-3] - 1) * 100 if len(close_prices) >= 3 else 0

    tier_key = tier.replace('Tier ', '').replace('.', '_')
    if f'tier{tier_key}_bb_dev' not in params: tier_key = '4'

    price_attractive = (stock_info.get('bb_lower_dev', 1) <= params[f'tier{tier_key}_bb_dev']) or \
                       (stock_info.get('deviation', 1) <= params[f'tier{tier_key}_ma_dev'])
    energy_condensed = stock_info.get('rsi', 100) <= params[f'tier{tier_key}_rsi']
    market_agreed = stock_info.get('volume_ratio', 0) >= params[f'tier{tier_key}_vol']
    macd_cross_occurred = stock_info.get('macd_cross', False)

    is_watching = price_attractive and energy_condensed
    is_captured = is_watching and market_agreed
    if tier in ['Tier 1.5', 'Tier 2']:
        is_watching = is_watching and macd_cross_occurred
        is_captured = is_watching and market_agreed

    status, status_order = "⚪️ 안정", 5
    if is_captured: status, status_order = "🟢 포착", 1
    elif is_watching: status, status_order = "🟡 감시", 2
    if price_change_rate <= -7 and status == "⚪️ 안정": status, status_order = "⚡ 변동성", 3
    if price_change_rate >= 7: status, status_order = "⚠️ 과열", 4
    
    price_score = (20 if stock_info.get('bb_lower_dev', 1) <= params[f'tier{tier_key}_bb_dev'] else 10 if stock_info.get('bb_lower_dev', 1) <= 0 else 0) + \
                  (20 if stock_info.get('deviation', 1) <= params[f'tier{tier_key}_ma_dev'] else 10 if stock_info.get('deviation', 1) <= 0 else 0)
    energy_score = 30 if stock_info.get('rsi', 100) <= params[f'tier{tier_key}_rsi'] else 15 if stock_info.get('rsi', 100) <= 50 else 0
    trend_score = 20 if macd_cross_occurred else 10 if stock_info.get('macd_latest', 0) > stock_info.get('signal_latest', 0) else 0
    agreement_score = 10 if market_agreed else 0
    total_score = price_score + energy_score + trend_score + agreement_score

    return {
        '상태': status, '종목명': stock_info['name'], '티커': stock_info['ticker'], '티어': tier,
        '3일 변동률': f"{price_change_rate:.1f}%",
        '가격 매력도': f"{'✅' if price_attractive else '❌'} (BB:{stock_info.get('bb_lower_dev', 0):.1f}%, MA:{stock_info.get('deviation', 0):.1f}%)",
        '에너지 응축': f"{'✅' if energy_condensed else '❌'} (RSI:{stock_info.get('rsi', 0):.1f})",
        '추세 전환': f"{'✅' if macd_cross_occurred else '❌'} ({'상승' if stock_info.get('macd_latest', 0) > stock_info.get('signal_latest', 0) else '하락'})",
        '시장 동의': f"{'✅' if market_agreed else '❌'} (거래량:{stock_info.get('volume_ratio', 0):.1f}배)",
        'status_order': status_order, '종합 점수': total_score,
        '상태_표시': f"{status} ({total_score}점)",
        '현재가': f"{stock_info['currency_symbol']}{stock_info['current_price']:.2f}",
        '기준일': stock_info['last_update_date'],
    }

# ==============================================================================
# --- 6. 데이터 로딩 및 유틸리티 함수 ---
# ==============================================================================
def get_currency_info(ticker):
    """티커를 기반으로 통화 정보(코드, 심볼)를 반환합니다."""
    return ("KRW", "₩") if ".KS" in ticker.upper() or ".KQ" in ticker.upper() else ("USD", "$")

@st.cache_data(ttl=300)
def get_stock_full_data(ticker, stock_name):
    """yfinance를 통해 주식 데이터를 가져오고 모든 지표를 계산합니다."""
    data = yf.Ticker(ticker).history(period="1y")
    if data is None or data.empty: return None

    close_prices = data['Close']
    volumes = data['Volume']
    macd_line, signal_line = calculate_macd(close_prices)
    _, ma20, lower_bb = calculate_bollinger_bands(close_prices)
    _, currency_symbol = get_currency_info(ticker)

    return {
        'name': stock_name, 'ticker': ticker, 'close_prices': close_prices,
        'current_price': close_prices.iloc[-1],
        'last_update_date': close_prices.index[-1].strftime('%Y-%m-%d'),
        'currency_symbol': currency_symbol,
        'deviation': ((close_prices.iloc[-1] / ma20.iloc[-1]) - 1) * 100 if not ma20.empty and ma20.iloc[-1] > 0 else 0,
        'bb_lower_dev': ((close_prices.iloc[-1] / lower_bb.iloc[-1]) - 1) * 100 if not lower_bb.empty and lower_bb.iloc[-1] > 0 else 0,
        'rsi': calculate_rsi(close_prices).iloc[-1] if not close_prices.empty else 50,
        'macd_cross': (macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] < signal_line.iloc[-2]) if len(macd_line) > 1 and len(signal_line) > 1 else False,
        'macd_latest': macd_line.iloc[-1] if not macd_line.empty else 0,
        'signal_latest': signal_line.iloc[-1] if not signal_line.empty else 0,
        'volume_ratio': (volumes.iloc[-1] / volumes.rolling(window=20).mean().iloc[-1]) if len(volumes) > 20 and volumes.rolling(window=20).mean().iloc[-1] > 0 else 1.0
    }

def clean_and_validate_df(df):
    if df is None or df.empty: return None, None
    df.columns = [str(col).strip().lower() for col in df.columns]
    column_mapping = {
        'ticker': ['ticker', '종목코드'], 'name': ['name', '종목명'], 
        'tier': ['tier', '티어', '자산티어'], 'value': ['현재평가금액'],
        'target_pct': ['목표비중(%)']
    }
    df.rename(columns={v: k for k, vs in column_mapping.items() for v in vs if v in df.columns}, inplace=True)
    if not all(col in df.columns for col in ['ticker', 'name', 'tier']): return None, None
    df.dropna(subset=['tier', 'ticker'], inplace=True)
    analysis_target_df = df[df['tier'].str.contains('Tier', na=False)].copy()
    return df, analysis_target_df

@st.cache_data(ttl=300)
def load_portfolio_from_sheets(sheet_key):
    try:
        gc = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
        worksheet = gc.open_by_key(sheet_key).sheet1
        full_df = get_as_dataframe(worksheet, evaluate_formulas=True).dropna(how='all').reset_index(drop=True)
        return clean_and_validate_df(full_df.copy())
    except Exception as e:
        st.error(f"Google Sheets 데이터 로드 중 오류: {e}")
        return None, None

@st.cache_data
def run_analysis_pipeline(_df, params):
    if _df is None or _df.empty: return pd.DataFrame(), {}
    all_results, ticker_map = [], {}
    progress_bar = st.progress(0, "분석 시작...")
    for i, row in _df.iterrows():
        progress_bar.progress((i + 1) / len(_df), f"분석 중: {row['name']}")
        ticker, stock_info = str(row['ticker']).strip().upper(), None
        final_ticker = None
        if ticker.isdigit():
            for suffix in ['.KS', '.KQ']:
                final_ticker = f"{ticker.zfill(6)}{suffix}"
                stock_info = get_stock_full_data(final_ticker, row['name'])
                if stock_info: break
        else:
            final_ticker = ticker
            stock_info = get_stock_full_data(final_ticker, row['name'])

        if stock_info:
            ticker_map[row['name']] = final_ticker
            res = analyze_stock(stock_info, row['tier'], params)
            if res: all_results.append(res)
    progress_bar.empty()
    return pd.DataFrame(all_results), ticker_map

# ==============================================================================
# --- 7. UI 컴포넌트 및 대시보드 함수 ---
# ==============================================================================
def display_macro_risk_dashboard():
    st.markdown("#### 🌍 글로벌 매크로 리스크 대시보드")
    indicators = get_macro_indicators()
    
    geo_risk = 4 
    geo_risk_reason = "중동 및 유럽 지역 분쟁 지속"
    
    total_score, status, scores = calculate_risk_score(indicators, geo_risk)
    
    color = "red" if status == "위험" else "orange" if status == "경계" else "green"
    
    col1, col2 = st.columns(2)
    col1.metric("종합 리스크 점수", f"{total_score:.1f} / 100")
    col2.markdown(f"<h2 style='color: {color}; text-align: right;'>상태: {status}</h2>", unsafe_allow_html=True)

    with st.expander("상세 리스크 지표 보기"):
        risk_data = {
            '구분': ['주식 시장', '주식 시장', '주식 시장', '통화/채권', '통화/채권', '통화/채권', '원자재', '원자재', '지정학'],
            '지표명': ['VIX', '공포&탐욕 지수', 'Put/Call 비율', '달러 인덱스(DXY)', '美 10년물 금리', '장단기 금리차', 'WTI 유가', '구리 가격', '주관적 판단'],
            '현재 값': [f"{indicators.get('VIX', 0):.2f}", f"{indicators.get('Fear_Greed', 0)}", f"{indicators.get('Put_Call_Ratio', 0):.2f}",
                      f"{indicators.get('DXY', 0):.2f}", f"{indicators.get('US10Y', 0):.2f}%", f"{indicators.get('Yield_Spread', 0):.2f}%",
                      f"${indicators.get('WTI', 0):.2f}", f"${indicators.get('Copper', 0):.2f}", f"{geo_risk}/10 ({geo_risk_reason})"],
            '리스크 점수': [f"{scores.get('VIX',0):.1f}", f"{scores.get('Fear_Greed',0):.1f}", f"{scores.get('Put_Call_Ratio',0):.1f}",
                        f"{scores.get('DXY',0):.1f}", f"{scores.get('US10Y',0):.1f}", f"{scores.get('Yield_Spread',0):.1f}",
                        f"{scores.get('WTI',0):.1f}", f"{scores.get('Copper',0):.1f}", f"{scores.get('Geo_Risk',0):.1f}"]
        }
        risk_df = pd.DataFrame(risk_data)

        def color_risk(val):
            score = float(val)
            color = 'red' if score > 70 else 'orange' if score > 40 else 'green'
            return f'color: {color}'
        
        st.dataframe(risk_df.style.map(color_risk, subset=['리스크 점수']), use_container_width=True, hide_index=True)

    st.markdown("---")
    return total_score, status

def create_main_dashboard(radar_df):
    st.header("📈 포트폴리오 분석")
    
    st.markdown("##### 📊 분석 파라미터")
    selected_sensitivity = st.radio("감시 민감도", ['엄격', '중간', '관대'], index=1, horizontal=True)

    params_config = {
        '엄격': {'bb_dev': -3, 'ma_dev': -10, 'rsi': 35, 'vol': 1.5},
        '중간': {'bb_dev': -2, 'ma_dev': -6, 'rsi': 40, 'vol': 1.2},
        '관대': {'bb_dev': -1, 'ma_dev': -5, 'rsi': 45, 'vol': 1.0}
    }
    base_params = params_config[selected_sensitivity]
    params = {}
    for t in ['1', '1_5', '2', '4']:
        multiplier = {'1': 1.0, '1_5': 1.2, '2': 1.5, '4': 1.1}[t]
        for k, v in base_params.items():
            params[f'tier{t}_{k}'] = v * multiplier if k != 'vol' else v
    
    analysis_df = st.session_state.get('analysis_target_data')
    if analysis_df is not None:
        radar_df, ticker_map = run_analysis_pipeline(analysis_df, params)
        st.session_state['ticker_map'] = ticker_map
        st.session_state['radar_results'] = radar_df

    st.subheader("📡 레이더 스크리닝 결과")
    sorted_df = radar_df.sort_values(by=['status_order', '종합 점수'], ascending=[True, False])
    display_cols = ['상태_표시', '종목명', '티어', '현재가', '기준일', '3일 변동률', '가격 매력도', '에너지 응축', '추세 전환', '시장 동의']
    st.dataframe(sorted_df[display_cols], use_container_width=True, hide_index=True)
    return sorted_df

def create_rebalancing_tab(full_df):
    st.header("⚖️ 포트폴리오 리밸런싱")
    
    if full_df is None or full_df.empty:
        st.warning("포트폴리오 데이터가 없습니다.")
        return

    full_df['value'] = pd.to_numeric(full_df.get('value'), errors='coerce').fillna(0)
    asset_df = full_df[full_df['tier'].str.contains('Tier', na=False)].copy()
    
    # Tier 4 제외
    asset_df = asset_df[asset_df['tier'] != 'Tier 4']

    if 'target_pct' in asset_df.columns:
        asset_df['target_pct'] = pd.to_numeric(asset_df['target_pct'], errors='coerce').fillna(0)
    else:
        st.warning("Google Sheets에 '목표비중(%)' 컬럼이 없습니다.")
        return

    total_asset_value = asset_df['value'].sum()

    if total_asset_value > 0:
        current_allocations = asset_df.groupby('tier')['value'].sum() / total_asset_value * 100
        target_allocations = asset_df.groupby('tier')['target_pct'].first()
        
        alloc_df = pd.DataFrame(target_allocations).rename(columns={'target_pct': '목표 비중'})
        alloc_df['현재 비중'] = alloc_df.index.map(current_allocations).fillna(0)
        
        # 티어 순서대로 정렬
        alloc_df['sort_key'] = alloc_df.index.str.replace('Tier ', '').astype(float)
        alloc_df = alloc_df.sort_values('sort_key').drop(columns=['sort_key'])
        alloc_df.reset_index(inplace=True)
        
        fig = go.Figure()
        
        # 겹쳐진 막대그래프 로직 (범례 중복 제거 및 색상/테두리 개선)
        # 더 큰 값을 뒤에, 작은 값을 앞에 그리기 위해 데이터 분리
        df_target_larger = alloc_df[alloc_df['목표 비중'] >= alloc_df['현재 비중']]
        df_current_larger = alloc_df[alloc_df['목표 비중'] < alloc_df['현재 비중']]

        # 목표가 크거나 같은 경우: 목표(뒤) -> 현재(앞)
        fig.add_trace(go.Bar(name='목표 비중', x=df_target_larger['tier'], y=df_target_larger['목표 비중'],
                           marker_color='lightgray', marker_line=dict(color='black', width=1),
                           legendgroup='target', showlegend=True))
        fig.add_trace(go.Bar(name='현재 비중', x=df_target_larger['tier'], y=df_target_larger['현재 비중'],
                           marker_color='steelblue', marker_line=dict(color='black', width=1),
                           legendgroup='current', showlegend=True))

        # 현재가 더 큰 경우: 현재(뒤) -> 목표(앞)
        fig.add_trace(go.Bar(name='현재 비중', x=df_current_larger['tier'], y=df_current_larger['현재 비중'],
                           marker_color='steelblue', marker_line=dict(color='black', width=1),
                           legendgroup='current', showlegend=False))
        fig.add_trace(go.Bar(name='목표 비중', x=df_current_larger['tier'], y=df_current_larger['목표 비중'],
                           marker_color='lightgray', marker_line=dict(color='black', width=1),
                           legendgroup='target', showlegend=False))


        fig.update_layout(barmode='overlay', title_text='목표 vs 현재 자산 배분', 
                          yaxis_title='비중 (%)', legend_traceorder="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("분석할 투자 자산이 없습니다.")

# ==============================================================================
# --- 8. 메인 실행 로직 ---
# ==============================================================================
def main():
    st.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; margin: 0;'>🎯 ROgicX 작전 본부 v17.0</h1>
        <p style='color: white; margin: 5px 0 0 0; font-size: 16px;'>핵심 임무 중심의 2차 판단 지원 시스템</p>
    </div>
    """, unsafe_allow_html=True)

    SHEET_KEY = "1AG2QrAlcjksI2CWp_6IuL5jCrFhzpOGl7casHvFGvi8"
    
    top_cols = st.columns([0.7, 0.15, 0.15])
    with top_cols[1]:
        if st.button("📝 브리핑 노트 생성"):
            st.session_state['show_briefing_note'] = True
    with top_cols[2]:
        if st.button("🔄 새로고침"):
            st.cache_data.clear()
            st.rerun()

    risk_score, risk_status = display_macro_risk_dashboard()
    st.session_state['risk_score'] = risk_score
    st.session_state['risk_status'] = risk_status

    if 'analysis_target_data' not in st.session_state:
        full_df, analysis_df = load_portfolio_from_sheets(SHEET_KEY)
        st.session_state['full_portfolio_data'] = full_df
        st.session_state['analysis_target_data'] = analysis_df
    else:
        full_df = st.session_state.get('full_portfolio_data')
        analysis_df = st.session_state.get('analysis_target_data')

    tab1, tab2 = st.tabs(["📊 포트폴리오 분석", "⚖️ 포트폴리오 리밸런싱"])

    with tab1:
        if analysis_df is not None and not analysis_df.empty:
            create_main_dashboard(pd.DataFrame())
        else:
            st.error("Google Sheets에서 데이터를 불러오지 못했거나 분석할 종목이 없습니다.")

    with tab2:
        create_rebalancing_tab(full_df)

    if st.session_state.get('show_briefing_note', False):
        st.markdown("---")
        radar_df = st.session_state.get('radar_results', pd.DataFrame())
        capture_count = len(radar_df[radar_df['상태'] == '🟢 포착']) if not radar_df.empty else 0
        watch_count = len(radar_df[radar_df['상태'] == '🟡 감시']) if not radar_df.empty else 0
        overheat_count = len(radar_df[radar_df['상태'] == '⚠️ 과열']) if not radar_df.empty else 0
        
        top_targets = radar_df[radar_df['status_order'] <= 2].head(2) if not radar_df.empty else pd.DataFrame()
        targets_text = ""
        for _, row in top_targets.iterrows():
            targets_text += f"        * `{row['종목명']}`: {row['상태']}, {row['종합 점수']}점.\n"
            targets_text += f"            * 가격 매력도: {row['가격 매력도']}\n"
            targets_text += f"            * 에너지 응축: {row['에너지 응축']}\n"
            targets_text += f"            * 추세 전환: {row['추세 전환']}\n"
            targets_text += f"            * 시장 동의: {row['시장 동의']}\n"

        note = f"""
**[ROgicX 작전 본부 브리핑 노트]**
* **일시:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
* **종합 리스크 점수:** {st.session_state.get('risk_score', 0):.1f}/100 (상태: {st.session_state.get('risk_status', 'N/A')})
* **레이더 스크리닝 요약:**
    * `🟢 포착`: {capture_count}개
    * `🟡 감시`: {watch_count}개
    * `⚠️ 과열`: {overheat_count}개
* **주요 논의 대상:**
{targets_text if not top_targets.empty else "        * (해당 없음)"}
"""
        st.text_area("생성된 브리핑 노트 (Ctrl+C로 복사)", note, height=250)
        if st.button("닫기"):
            st.session_state['show_briefing_note'] = False
            st.rerun()

if __name__ == "__main__":
    main()
