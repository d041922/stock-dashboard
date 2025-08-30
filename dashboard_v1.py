# dashboard_v17_revised.py

import streamlit as st
import finnhub
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import google.generativeai as genai


# --- 로그인 함수 ---
def login():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        password = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if password == st.secrets["password"]:
                st.session_state["authenticated"] = True
                st.success("Login successful ✅")
            else:
                st.error("Incorrect password 🚫")
        # 로그인 안되면 앱 나머지 실행 중단
        st.stop()

# --- 페이지 시작 ---
st.set_page_config(page_title="GEM Dashboard", page_icon="💎", layout="wide")

# 로그인 먼저 체크
login()

# 로그인 성공 시 표시되는 메인 화면
st.title("💎 GEM: Finance Dashboard")
st.write("메인 대시보드 콘텐츠 표시")

# --- API 키 설정 ---
try:
    FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"API 키를 secrets.toml 파일에 설정해주세요. 오류: {e}")
    st.stop()

# --- 2. 데이터 호출 함수 ---

@st.cache_data
def get_company_profile(ticker):
    """회사 프로필 정보를 가져옵니다."""
    return finnhub_client.company_profile2(symbol=ticker)

@st.cache_data
def get_company_news(ticker):
    """최근 30일간의 회사 뉴스를 가져옵니다."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    return finnhub_client.company_news(ticker, _from=start_date, to=end_date)

@st.cache_data
def get_quote(ticker):
    """실시간 시세 정보를 가져옵니다."""
    return finnhub_client.quote(ticker)

@st.cache_data
def get_basic_financials(ticker):
    """연간 기본 재무 데이터를 가져와 처리합니다."""
    try:
        financials = finnhub_client.company_basic_financials(ticker, 'all')
        if 'series' in financials and 'annual' in financials['series']:
            annual_data = financials['series']['annual']
            records = {}
            for metric, data_points in annual_data.items():
                if not data_points: continue
                clean_metric_name = metric.replace('Value', '').lower()
                for point in data_points:
                    period = point.get('period')
                    value = point.get('v')
                    if period not in records: records[period] = {}
                    records[period][clean_metric_name] = value
            if not records: return pd.DataFrame()
            df = pd.DataFrame.from_dict(records, orient='index').sort_index()
            df.columns = [col.lower() for col in df.columns]
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"기본 재무 정보 API 호출 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

@st.cache_data
def get_company_peers(ticker):
    """경쟁사 Ticker 목록을 가져옵니다."""
    try:
        return finnhub_client.company_peers(ticker)
    except Exception as e:
        st.warning(f"경쟁사 정보 API 호출 중 오류 발생: {e}")
        return []

@st.cache_data
def get_company_earnings(ticker):
    """과거 분기별 실적 발표 데이터를 가져옵니다."""
    try:
        earnings = finnhub_client.company_earnings(ticker, limit=5)
        if not earnings: return pd.DataFrame()
        df = pd.DataFrame(earnings)
        if 'surprisePercent' in df.columns:
            df['EPS 결과'] = df['surprisePercent'].apply(lambda x: 'Beat' if pd.notnull(x) and x > 0 else ('Miss' if pd.notnull(x) and x < 0 else 'Meet'))
        else:
            df['surprisePercent'], df['EPS 결과'] = None, 'N/A'
        final_cols_map = {'period': '발표 분기', 'actual': '실제 EPS', 'estimate': '예상 EPS', 'surprisePercent': 'EPS 서프라이즈 (%)', 'EPS 결과': 'EPS 결과'}
        existing_cols = [col for col in final_cols_map.keys() if col in df.columns]
        final_df = df[existing_cols].rename(columns=final_cols_map)
        return final_df
    except Exception as e:
        st.warning(f"실적 정보 API 호출 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data
def get_earnings_calendar(ticker):
    """다음 실적 발표일 정보를 가져옵니다."""
    try:
        today, one_year_later = datetime.now().strftime('%Y-%m-%d'), (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
        calendar = finnhub_client.earnings_calendar(_from=today, to=one_year_later, symbol=ticker)
        if calendar and 'earningsCalendar' in calendar and calendar['earningsCalendar']:
            return calendar['earningsCalendar'][0].get('date')
        return None
    except Exception as e:
        st.warning(f"다음 실적 발표일 API 호출 중 오류 발생: {e}")
        return None

@st.cache_data
def get_stock_candles(ticker):
    """yfinance를 사용하여 과거 1년간의 일봉 데이터를 가져옵니다."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty:
            st.warning(f"yfinance에서 '{ticker}'에 대한 차트 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"yfinance 데이터 호출 중 오류 발생: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    """데이터프레임에 이동평균선, RSI, 볼린저밴드, MACD를 추가합니다."""
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['UpperBand'] = df['SMA20'] + (df['StdDev'] * 2)
    df['LowerBand'] = df['SMA20'] - (df['StdDev'] * 2)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def generate_technical_summary(df):
    """기술적 지표를 바탕으로 분석 요약을 생성합니다."""
    summary = []
    if len(df) < 2: return ["데이터가 부족하여 분석할 수 없습니다."]
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    if latest['RSI14'] > 70: summary.append("📈 **RSI 과매수:** 현재 RSI가 70 이상으로, 주가가 과도하게 상승했을 수 있습니다. (단기 조정 가능성)")
    elif latest['RSI14'] < 30: summary.append("📉 **RSI 과매도:** 현재 RSI가 30 이하로, 주가가 과도하게 하락했을 수 있습니다. (단기 반등 가능성)")
    else: summary.append(f"📊 **RSI:** 현재 {latest['RSI14']:.2f}로 중립 구간에 있습니다.")
    if previous['SMA20'] < previous['SMA60'] and latest['SMA20'] > latest['SMA60']: summary.append("🚀 **골든 크로스 발생:** 단기(20일) 이동평균선이 장기(60일) 이동평균선을 상향 돌파했습니다. (강세 신호)")
    elif previous['SMA20'] > previous['SMA60'] and latest['SMA20'] < latest['SMA60']: summary.append("⚠️ **데드 크로스 발생:** 단기(20일) 이동평균선이 장기(60일) 이동평균선을 하향 돌파했습니다. (약세 신호)")
    if previous['MACD'] < previous['SignalLine'] and latest['MACD'] > latest['SignalLine']: summary.append("📈 **MACD 상향 돌파:** MACD 선이 시그널 선을 상향 돌파하여 매수 신호일 수 있습니다.")
    elif previous['MACD'] > previous['SignalLine'] and latest['MACD'] < latest['SignalLine']: summary.append("📉 **MACD 하향 돌파:** MACD 선이 시그널 선을 하향 돌파하여 매도 신호일 수 있습니다.")
    if latest['Close'] > latest['UpperBand']: summary.append("🖐️ **볼린저 밴드 상단 돌파:** 주가가 상단 밴드를 넘어섰습니다. 과매수 상태일 수 있습니다.")
    elif latest['Close'] < latest['LowerBand']: summary.append("🙌 **볼린저 밴드 하단 돌파:** 주가가 하단 밴드 아래로 내려갔습니다. 과매도 상태일 수 있습니다.")
    if not summary: summary.append("뚜렷한 기술적 신호가 없습니다.")
    return summary

@st.cache_data
def get_watchlist_summary_data(ticker_list):
    """주어진 Ticker 리스트에 대한 강화된 비교 데이터를 가져옵니다."""
    summary_data = []
    progress_bar = st.progress(0, text="관심종목 데이터 수집 중...")
    for i, ticker in enumerate(ticker_list):
        try:
            stock_info, quote = yf.Ticker(ticker).info, get_quote(ticker)
            current_price = stock_info.get('currentPrice', quote.get('c', 0))
            change_percent = (current_price / stock_info.get('previousClose', 1) - 1) * 100 if stock_info.get('previousClose') else quote.get('dp', 0)
            market_cap, volume = stock_info.get('marketCap', 0) / 1_000_000, stock_info.get('volume', 0)
            fifty_two_week_high, fifty_two_week_low = stock_info.get('fiftyTwoWeekHigh', 0), stock_info.get('fiftyTwoWeekLow', 0)
            candles_df = get_stock_candles(ticker)
            rsi = candles_df.iloc[-1]['RSI14'] if not candles_df.empty and 'RSI14' in add_technical_indicators(candles_df).columns else None
            summary_data.append({"Ticker": ticker, "Price": current_price, "% Change": change_percent, "RSI": rsi, "Volume": volume, "Market Cap (M)": market_cap, "52 Week High": fifty_two_week_high, "52 Week Low": fifty_two_week_low})
        except Exception: continue
        finally: progress_bar.progress((i + 1) / len(ticker_list), text=f"관심종목 데이터 수집 중: {ticker}")
    progress_bar.empty()
    return pd.DataFrame(summary_data)

@st.cache_data
def get_peer_summary(ticker_list):
    """경쟁사 비교를 위한 간단한 요약 데이터를 가져옵니다."""
    summary_data = []
    for ticker in ticker_list:
        try:
            profile, quote = get_company_profile(ticker), get_quote(ticker)
            summary_data.append({"Ticker": ticker, "Name": profile.get('name', ticker), "Market Cap (M)": profile.get('marketCapitalization', 0), "% Change": quote.get('dp', 0)})
        except: continue
    return pd.DataFrame(summary_data)

# [개선] Gemini AI 분석 함수를 스트리밍 방식으로 변경
def generate_gemini_analysis_stream(ticker, profile, quote, financials_df, tech_summary, news):
    """모든 데이터를 종합하여 Gemini AI에게 전략 분석을 스트리밍으로 요청합니다."""
    model = genai.GenerativeModel('gemini-2.5-pro')
    profile_info = f"회사명: {profile.get('name', 'N/A')}, 산업: {profile.get('finnhubIndustry', 'N/A')}, 시가총액(M): {profile.get('marketCapitalization', 'N/A'):,}"
    quote_info = f"현재가: ${quote.get('c', 0):.2f}, 변동률: {quote.get('dp', 0):.2f}%"
    financials_info = "최근 3년 재무 요약:\n" + financials_df.tail(3).to_string() if not financials_df.empty else "재무 데이터 없음"
    tech_info = "기술적 분석 요약:\n- " + "\n- ".join(tech_summary)
    news_headlines = "최신 뉴스 헤드라인:\n- " + "\n- ".join([item['headline'] for item in news[:5]]) if news else "최신 뉴스 없음"
    master_prompt = f"""
    **SYSTEM ROLE:** 당신은 월스트리트 최고의 금융 분석가이자, 'MASTER'라는 이름의 중장기 가치 투자자를 보좌하는 AI 전략 파트너입니다. 당신의 임무는 흩어진 데이터를 종합하여 명확하고 실행 가능한 투자 인사이트를 제공하는 것입니다.
    **MASTER의 투자 철학:** 좋은 종목을 좋은 시기에 좋은 가격에 매수하여 경제적 자유를 달성하는 것을 목표로 합니다. 단기적인 소음에 흔들리지 않고 기업의 본질 가치에 집중합니다.
    **분석 대상:** - Ticker: {ticker}
    **입력 데이터:**
    1. **기업 개요:** {profile_info}
    2. **현재 시세:** {quote_info}
    3. **핵심 재무 데이터 (연간):**\n{financials_info}
    4. **기술적 분석:**\n{tech_info}
    5. **최신 뉴스:**\n{news_headlines}
    **MISSION:** 위 모든 데이터를 종합적으로 분석하여, MASTER의 투자 철학에 입각한 전략 브리핑을 생성하십시오. 아래 4가지 핵심 질문에 대해 명확하고 논리적으로 답변해야 합니다. 답변은 한국어로, 전문가의 어조를 유지하되 이해하기 쉽게 작성하십시오.
    ---
    ### 💎 {ticker} 전략 브리핑
    #### 1. 좋은 종목인가? (What to Buy?)
    - 이 기업의 비즈니스 모델과 산업 내 위치를 평가하십시오.
    - 재무 건전성과 성장성을 분석하여 '좋은 기업'의 조건을 충족하는지 판단하십시오.
    #### 2. 좋은 시기인가? (When to Buy?)
    - 기술적 분석(골든/데드 크로스, RSI, MACD 등)을 통해 현재 시장의 심리와 추세를 평가하십시오.
    - 최신 뉴스가 주가에 미칠 단기적, 중기적 영향을 분석하십시오.
    #### 3. 좋은 가격인가? (What Price?)
    - 현재 주가가 기업의 내재 가치(재무 상태, 성장성) 대비 매력적인 수준인지 평가하십시오.
    - 기술적 지지/저항 수준을 고려하여 가격의 적정성을 판단하십시오.
    #### 4. 어떻게 행동해야 하는가? (How to Act?)
    - 위 3가지 분석을 종합하여 MASTER를 위한 최종적인 '액션 플랜'을 제안하십시오. (예: "현재는 긍정적인 신호가 많아 분할 매수를 고려할 만한 시점입니다.", "기술적 과열 신호가 있어, 추가적인 조정 시까지 관망하는 것이 유리합니다." 등)
    - 잠재적인 리스크와 기회 요인을 함께 제시하여 균형 잡힌 시각을 제공하십시오.
    """
    try:
        response = model.generate_content(master_prompt, stream=True)
        for chunk in response:
            yield chunk.text
    except Exception as e:
        yield f"Gemini 분석 중 오류가 발생했습니다: {e}"

# --- 3. UI 구성 및 4. 데이터 처리 ---
st.title("👑 GEM: Finance Dashboard")
st.caption("v25.0 - Streaming AI Analysis")

if 'tickers' not in st.session_state:
    st.session_state.tickers = []

with st.sidebar:
    st.header("Controls")
    tickers_input = st.text_area("Ticker(s)", value="NVDA, AAPL, MSFT, GOOGL", help="분석할 종목의 Ticker를 쉼표(,)나 줄바꿈으로 구분하여 입력하세요. 첫 번째 종목을 상세 분석합니다.")
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.clear()
        st.success("Cache and session state cleared!")
        st.rerun()

    if st.button("🔍 분석 실행", use_container_width=True, type="primary"):
        st.session_state.tickers = [ticker.strip().upper() for ticker in tickers_input.replace(',', '\n').split('\n') if ticker.strip()]

if st.session_state.tickers:
    tickers = st.session_state.tickers
    main_ticker = tickers[0]
        
    with st.spinner(f"'{main_ticker}' 및 관심종목 데이터를 가져오는 중..."):
        try:
            profile = get_company_profile(main_ticker)
            quote = get_quote(main_ticker)
            news = get_company_news(main_ticker)
            financials_df = get_basic_financials(main_ticker)
            peers = get_company_peers(main_ticker)
            earnings_data = get_company_earnings(main_ticker)
            next_earnings_date = get_earnings_calendar(main_ticker)
            candles_df = get_stock_candles(main_ticker)
            
            tab_names = ["💎 GEMINI 분석", "⭐️ 관심종목", "📊 개요", "📈 기술적 분석", "💰 재무", "👥 경쟁사 비교", "📈 실적", "📰 뉴스"]
            gemini_tab, watchlist_tab, overview_tab, tech_tab, fin_tab, peer_tab, earn_tab, news_tab = st.tabs(tab_names)

            with gemini_tab:
                st.subheader(f"💎 {main_ticker} AI 전략 분석")
                st.info("아래 버튼을 클릭하면 수집된 모든 데이터를 바탕으로 Gemini AI가 종합적인 전략 브리핑을 생성합니다.")
                
                if st.button("🚀 GEMINI 전략 분석 실행", use_container_width=True, type="primary"):
                    with st.spinner("AI가 데이터를 분석하고 전략을 수립하는 중..."):
                        tech_summary = []
                        if not candles_df.empty and len(candles_df) > 60:
                            candles_df_with_indicators = add_technical_indicators(candles_df.copy())
                            tech_summary = generate_technical_summary(candles_df_with_indicators)
                        
                        # [개선] st.write_stream을 사용하여 실시간으로 AI 응답 표시
                        st.write_stream(generate_gemini_analysis_stream(main_ticker, profile, quote, financials_df, tech_summary, news))

            with watchlist_tab:
                st.subheader("관심종목 요약 (Watchlist Summary)")
                if len(tickers) > 1:
                    watchlist_df = get_watchlist_summary_data(tickers)
                    if not watchlist_df.empty:
                        st.info("관심종목들의 실시간 현황과 핵심 기술/가치 지표를 한눈에 비교하여 가장 주목할 만한 종목을 빠르게 찾아낼 수 있습니다.")
                        st.dataframe(watchlist_df.set_index('Ticker').style.format({"Price": "${:,.2f}", "% Change": "{:.2f}%", "RSI": "{:.2f}", "Volume": "{:,.0f}", "Market Cap (M)": "{:,.0f}", "52 Week High": "${:,.2f}", "52 Week Low": "${:,.2f}"}, na_rep="-").background_gradient(cmap='RdYlGn', subset=['% Change']))
                    else: st.warning("관심종목에 대한 요약 정보를 가져오는 데 실패했습니다.")
                else: st.info("여러 종목을 입력하시면 이곳에서 비교 분석표를 확인할 수 있습니다.")

            with overview_tab:
                if profile:
                    st.subheader(f"{profile.get('name', main_ticker)} ({main_ticker})")
                    col1, col2 = st.columns([1, 4])
                    with col1: st.image(profile.get('logo'), width=100)
                    with col2:
                        st.text(f"Industry: {profile.get('finnhubIndustry')}")
                        st.text(f"Market Cap: {profile.get('marketCapitalization', 0):,} M")
                        st.link_button("Visit Website", profile.get('weburl'))
                else: st.subheader(f"📈 {main_ticker} (프로필 정보 없음)")
                st.divider()
                st.subheader("실시간 주가 정보")
                if quote and quote.get('c') != 0:
                    cols = st.columns(4)
                    cols[0].metric("현재가", f"${quote.get('c', 0):.2f}", f"{quote.get('d', 0):.2f}$ ({quote.get('dp', 0):.2f}%)")
                    cols[1].metric("시가", f"${quote.get('o', 0):.2f}")
                    cols[2].metric("고가", f"${quote.get('h', 0):.2f}")
                    cols[3].metric("저가", f"${quote.get('l', 0):.2f}")
                else: st.warning("실시간 시세 정보를 가져올 수 없습니다.")
                st.divider()
                if next_earnings_date:
                    st.subheader("다가오는 주요 이벤트")
                    st.info(f"**다음 실적 발표 예정일:** {next_earnings_date}")

            with tech_tab:
                st.subheader("주가 차트 및 기술적 지표")
                if not candles_df.empty and len(candles_df) > 60:
                    candles_df = add_technical_indicators(candles_df)
                    st.subheader("기술적 분석 요약")
                    tech_summary = generate_technical_summary(candles_df)
                    for point in tech_summary: st.markdown(f"- {point}")
                    st.divider()
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Candlestick & Bollinger Bands', 'MACD', 'RSI'), row_heights=[0.6, 0.2, 0.2])
                    fig.add_trace(go.Candlestick(x=candles_df['Date'], open=candles_df['Open'], high=candles_df['High'], low=candles_df['Low'], close=candles_df['Close'], name='Price'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=candles_df['Date'], y=candles_df['UpperBand'], name='Upper Band', line=dict(color='rgba(152, 202, 255, 0.5)', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=candles_df['Date'], y=candles_df['LowerBand'], name='Lower Band', line=dict(color='rgba(152, 202, 255, 0.5)', width=1), fill='tonexty', fillcolor='rgba(152, 202, 255, 0.1)'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=candles_df['Date'], y=candles_df['SMA20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=candles_df['Date'], y=candles_df['SMA60'], name='SMA 60', line=dict(color='purple', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=candles_df['Date'], y=candles_df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=candles_df['Date'], y=candles_df['SignalLine'], name='Signal Line', line=dict(color='red', width=1)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=candles_df['Date'], y=candles_df['RSI14'], name='RSI 14', line=dict(color='royalblue', width=1)), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                    fig.update_layout(height=800, xaxis_rangeslider_visible=False, legend_orientation="h", legend=dict(y=1.1, x=0.5, xanchor="center"))
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info(f"'{main_ticker}'에 대한 차트 데이터를 가져오거나 분석하기에 데이터가 부족합니다.")
            
            with fin_tab:
                display_financials_df = financials_df.copy()
                if not display_financials_df.empty: display_financials_df.columns = [col.capitalize() for col in display_financials_df.columns]
                st.subheader("핵심 재무 지표 (연간, 단위: 백만)")
                if not display_financials_df.empty:
                    st.dataframe(display_financials_df.style.format("{:,.2f}", na_rep="-"))
                    for col in display_financials_df.columns:
                        if display_financials_df[col].notna().any():
                            fig = px.bar(display_financials_df.dropna(subset=[col]), y=col, title=f"Annual {col} (M)", labels={"value": f"{col} (M)", "index": "Year"})
                            st.plotly_chart(fig, use_container_width=True)
                else: st.warning(f"'{main_ticker}'에 대한 기본 재무 데이터를 찾을 수 없습니다.")

            with peer_tab:
                st.subheader(f"경쟁사 핵심 지표 비교")
                all_tickers_for_comparison = list(dict.fromkeys([main_ticker] + peers))[:6] if peers else [main_ticker]
                peer_df = get_peer_summary(all_tickers_for_comparison)
                if not peer_df.empty:
                    st.info("핵심 지표를 통해 경쟁사들과의 위치를 한눈에 비교할 수 있습니다.")
                    st.dataframe(peer_df.set_index('Ticker').style.format({"Market Cap (M)": "{:,.0f}", "% Change": "{:.2f}%"}, na_rep="-").background_gradient(cmap='RdYlGn', subset=['% Change']))
                else: st.warning("경쟁사 정보를 가져올 수 없습니다.")

            with earn_tab:
                st.subheader("분기별 실적 발표 내역")
                if not earnings_data.empty:
                    st.info("과거 실적 발표 내역을 통해 EPS(주당순이익)의 추이를 파악할 수 있습니다.")
                    st.dataframe(earnings_data.style.format({"실제 EPS": "{:.2f}", "예상 EPS": "{:.2f}", "EPS 서프라이즈 (%)": "{:.2f}%"}, na_rep="-"), use_container_width=True)
                    if 'EPS 서프라이즈 (%)' in earnings_data.columns:
                        fig_eps = px.bar(earnings_data.dropna(subset=['EPS 서프라이즈 (%)']), x='발표 분기', y='EPS 서프라이즈 (%)', title='분기별 EPS 서프라이즈 (%)', color='EPS 결과', color_discrete_map={'Beat': 'green', 'Miss': 'red', 'Meet': 'blue'})
                        fig_eps.add_hline(y=0)
                        st.plotly_chart(fig_eps, use_container_width=True)
                else: st.warning(f"'{main_ticker}'에 대한 실적 발표 내역이 없습니다.")
            
            with news_tab:
                st.subheader("최신 관련 뉴스")
                if news:
                    for item in news[:10]:
                        news_date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M')
                        st.markdown(f"**[{item['headline']}]({item['url']})**\n- *Source: {item['source']} | {news_date}*")
                else: st.info("관련 뉴스가 없습니다.")

        except Exception as e:
            st.error(f"알 수 없는 오류가 발생했습니다: {e}")

else:
    st.info("사이드바에서 Ticker를 입력하고 '분석 실행' 버튼을 클릭하세요.")

