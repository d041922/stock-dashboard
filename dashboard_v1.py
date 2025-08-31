# dashboard_v34_final.py

import streamlit as st
import finnhub
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
import json

# --- 1. 초기 설정 및 로그인 ---
st.set_page_config(page_title="GEM: Finance Dashboard", page_icon="💎", layout="wide")

def login():
    if "authenticated" not in st.session_state: st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        st.title("💎 GEM: Finance Dashboard")
        st.caption("MASTER, please enter the password to access the command center.")
        password = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if "password" in st.secrets and password == st.secrets["password"]:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password or password not set in secrets.toml 🚫")
        st.stop()

login()

# --- 2. API 키 및 인증 설정 ---
try:
    FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    gc = gspread.authorize(creds)
    SPREADSHEET_NAME = "GEM_Finance_Portfolio"
except Exception as e:
    st.error(f"API 키 또는 인증 정보를 secrets.toml 파일에 설정해주세요. 오류: {e}")
    st.stop()

# --- 3. 데이터 호출 및 처리 함수 ---

def load_data_from_gsheet():
    try:
        spreadsheet = gc.open(SPREADSHEET_NAME)
        portfolio_ws = spreadsheet.worksheet("Portfolio")
        portfolio_data = portfolio_ws.get_all_values()
        portfolio_headers = portfolio_data.pop(0)
        portfolio_df = pd.DataFrame(portfolio_data, columns=portfolio_headers)

        watchlist_ws = spreadsheet.worksheet("Watchlist")
        watchlist_df = pd.DataFrame(watchlist_ws.get_all_records())

        cash_ws = spreadsheet.worksheet("Cash")
        cash_df = pd.DataFrame(cash_ws.get_all_records())

        numeric_cols = ['수량', '평균 단가(USD)', '평균 단가(KRW)', '수동 현재가(KRW)']
        for col in numeric_cols:
            if col in portfolio_df.columns:
                portfolio_df[col] = portfolio_df[col].replace('', '0').astype(str).str.replace(',', '')
                portfolio_df[col] = pd.to_numeric(portfolio_df[col], errors='coerce').fillna(0)
        
        if '수동 수익률(%)' in portfolio_df.columns:
            portfolio_df['수동 수익률(%)'] = portfolio_df['수동 수익률(%)'].replace('', '0').astype(str).str.replace('%', '')
            portfolio_df['수동 수익률(%)'] = pd.to_numeric(portfolio_df['수동 수익률(%)'], errors='coerce').fillna(0) / 100
        
        if '금액(KRW)' in cash_df.columns:
            cash_df['금액(KRW)'] = cash_df['금액(KRW)'].replace('', '0').astype(str).str.replace(',', '')
            cash_df['금액(KRW)'] = pd.to_numeric(cash_df['금액(KRW)'], errors='coerce').fillna(0)

        return portfolio_df, watchlist_df, cash_df
    except gspread.exceptions.WorksheetNotFound as e:
        st.error(f"Google Sheets에서 '{e.args[0]}' 시트를 찾을 수 없습니다. ('Portfolio', 'Watchlist', 'Cash', 'Analysis_Log')")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"Google Sheets 로딩 중 오류: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# [누락된 함수 추가] AI 분석 전용 현금 데이터 로더
def load_cash_data_only():
    try:
        spreadsheet = gc.open(SPREADSHEET_NAME)
        cash_ws = spreadsheet.worksheet("Cash")
        cash_df = pd.DataFrame(cash_ws.get_all_records())
        if '금액(KRW)' in cash_df.columns:
            # 금액(KRW) 열을 문자열로 변환하고, 빈 문자열과 쉼표를 처리
            cash_df['금액(KRW)'] = cash_df['금액(KRW)'].astype(str).str.replace(',', '').replace('', '0')
            # 숫자 형식으로 최종 변환
            cash_df['금액(KRW)'] = pd.to_numeric(cash_df['금액(KRW)'], errors='coerce').fillna(0)
        return cash_df
    except Exception as e:
        st.error(f"Cash 시트 로딩 중 오류 발생: {e}")
        return pd.DataFrame()


def save_analysis_to_gsheet(log_data):
    try:
        spreadsheet = gc.open(SPREADSHEET_NAME)
        log_ws = spreadsheet.worksheet("Analysis_Log")
        EXPECTED_HEADERS = ["Timestamp", "종목코드", "AI_Model", "당시 주가", "분석 요약", "전체 분석 내용", "주요 데이터"]
        if not log_ws.get_all_values():
            log_ws.append_row(EXPECTED_HEADERS)
        row_to_append = [log_data.get(h, "") for h in EXPECTED_HEADERS]
        log_ws.append_row(row_to_append)
        return True
    except Exception as e:
        st.error(f"분석 기록 저장 중 오류 발생: {e}")
        return False

@st.cache_data(ttl=1)
def load_analysis_log(ticker):
    try:
        spreadsheet = gc.open(SPREADSHEET_NAME)
        log_ws = spreadsheet.worksheet("Analysis_Log")
        log_data = log_ws.get_all_records()
        if not log_data:
            return pd.DataFrame()
        log_df = pd.DataFrame(log_data)
        if '종목코드' not in log_df.columns:
            return pd.DataFrame()
        return log_df[log_df['종목코드'] == ticker].sort_values(by='Timestamp', ascending=False)
    except Exception as e:
        st.warning(f"과거 분석 기록 로딩 중 오류: {e}")
        return pd.DataFrame()

@st.cache_data
def get_company_profile(ticker): return finnhub_client.company_profile2(symbol=ticker)
@st.cache_data
def get_company_news(ticker):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    return finnhub_client.company_news(ticker, _from=start_date, to=end_date)
@st.cache_data
def get_quote(ticker): return finnhub_client.quote(ticker)
@st.cache_data
def get_basic_financials(ticker):
    try:
        financials = finnhub_client.company_basic_financials(ticker, 'all')
        if 'series' in financials and 'annual' in financials['series']:
            records = {}
            for metric, data_points in financials['series']['annual'].items():
                if not data_points: continue
                clean_metric = metric.replace('Value', '').lower()
                for point in data_points:
                    period, value = point.get('period'), point.get('v')
                    if period not in records: records[period] = {}
                    records[period][clean_metric] = value
            if not records: return pd.DataFrame()
            df = pd.DataFrame.from_dict(records, orient='index').sort_index()
            return df
        return pd.DataFrame()
    except Exception: return pd.DataFrame()
@st.cache_data
def get_company_peers(ticker):
    try: return finnhub_client.company_peers(ticker)
    except Exception: return []
@st.cache_data
def get_company_earnings(ticker):
    try:
        earnings = finnhub_client.company_earnings(ticker, limit=5)
        if not earnings: return pd.DataFrame()
        df = pd.DataFrame(earnings)
        if 'surprisePercent' in df.columns:
            df['EPS 결과'] = df['surprisePercent'].apply(lambda x: 'Beat' if pd.notnull(x) and x > 0 else 'Miss' if pd.notnull(x) and x < 0 else 'Meet')
        else: df['surprisePercent'], df['EPS 결과'] = None, 'N/A'
        final_cols = {'period': '발표 분기', 'actual': '실제 EPS', 'estimate': '예상 EPS', 'surprisePercent': 'EPS 서프라이즈 (%)', 'EPS 결과': 'EPS 결과'}
        return df[[c for c in final_cols if c in df.columns]].rename(columns=final_cols)
    except Exception: return pd.DataFrame()
@st.cache_data
def get_earnings_calendar(ticker):
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        later = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
        calendar = finnhub_client.earnings_calendar(_from=today, to=later, symbol=ticker)
        if calendar and calendar.get('earningsCalendar'): return calendar['earningsCalendar'][0].get('date')
        return None
    except Exception: return None
@st.cache_data
def get_stock_candles(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1y")
        if df.empty: return pd.DataFrame()
        return df.reset_index()
    except Exception: return pd.DataFrame()
def add_technical_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df
def generate_technical_summary(df):
    summary = []
    if len(df) < 60: return ["데이터가 부족하여 분석할 수 없습니다."]
    latest = df.iloc[-1]; previous = df.iloc[-2]
    if latest['RSI14'] > 70: summary.append("📈 **RSI 과매수:** 단기 조정 가능성.")
    elif latest['RSI14'] < 30: summary.append("📉 **RSI 과매도:** 단기 반등 가능성.")
    else: summary.append(f"📊 **RSI:** {latest['RSI14']:.2f} (중립)")
    if previous['SMA20'] < previous['SMA60'] and latest['SMA20'] > latest['SMA60']: summary.append("🚀 **골든 크로스:** 강세 신호.")
    elif previous['SMA20'] > previous['SMA60'] and latest['SMA20'] < latest['SMA60']: summary.append("⚠️ **데드 크로스:** 약세 신호.")
    if previous['MACD'] < previous['SignalLine'] and latest['MACD'] > latest['SignalLine']: summary.append("📈 **MACD 상향 돌파:** 매수 신호.")
    elif previous['MACD'] > previous['SignalLine'] and latest['MACD'] < latest['SignalLine']: summary.append("📉 **MACD 하향 돌파:** 매도 신호.")
    if not summary: summary.append("뚜렷한 기술적 신호가 없습니다.")
    return summary
def stream_and_capture_analysis(ticker, profile, quote, financials_df, tech_summary, news, portfolio_context):
    model_name = 'gemini-2.5-pro'
    model = genai.GenerativeModel(model_name)
    master_prompt = f"""
    **SYSTEM ROLE:** 당신은 월스트리트 최고의 금융 분석가이자, 'MASTER'라는 투자자를 보좌하는 AI 전략 파트너입니다.
    **MASTER의 투자 철학:** 좋은 종목을 좋은 시기에 좋은 가격에 매수하여 중장기적으로 부를 축적합니다.
    **분석 대상:** {ticker}
    **MASTER의 현재 포트폴리오 상황 (종목 보유량 + 현금 잔고 반드시 고려):**
    {portfolio_context}
    **입력 데이터:**
    1. **기업 개요:** 회사명: {profile.get('name', 'N/A')}, 산업: {profile.get('finnhubIndustry', 'N/A')}, 시총(M): {profile.get('marketCapitalization', 'N/A'):,}
    2. **현재 시세:** 현재가: ${quote.get('c', 0):.2f}, 변동률: {quote.get('dp', 0):.2f}%
    3. **핵심 재무:**\n{financials_df.tail(3).to_string() if not financials_df.empty else "재무 데이터 없음"}
    4. **기술적 분석:**\n- {"\n- ".join(tech_summary)}
    5. **최신 뉴스:**\n- {"\n- ".join([item['headline'] for item in news[:5]]) if news else "최신 뉴스 없음"}
    **MISSION:** 위 모든 데이터를 종합하여, MASTER의 투자 철학에 입각한 전략 브리핑을 생성하십시오. 아래 4가지 핵심 질문에 대해 명확하고 논리적으로 답변해야 합니다.
    특히, '4. 어떻게 행동해야 하는가?' 항목에서는 **MASTER의 현재 포트폴리오 상황을 반드시 고려하여**, 구체적이고 개인화된 액션 플랜을 제안해야 합니다.
    ---
    ### 💎 {ticker} 전략 브리핑
    *Analysis Model: `{model_name}`*
    #### 1. 좋은 종목인가? (What to Buy?)
    #### 2. 좋은 시기인가? (When to Buy?)
    #### 3. 좋은 가격인가? (What Price?)
    #### 4. 어떻게 행동해야 하는가? (How to Act?)
    """
    full_response = []
    try:
        response = model.generate_content(master_prompt, stream=True)
        for chunk in response:
            full_response.append(chunk.text)
            yield chunk.text
        st.session_state.last_analysis_text = "".join(full_response)
        st.session_state.last_model_used = model_name
    except Exception as e:
        error_message = f"Gemini 분석 중 오류가 발생했습니다: {e}"
        st.session_state.last_analysis_text = error_message
        st.session_state.last_model_used = "Error"
        yield error_message
@st.cache_data(ttl=300)
def get_current_prices_and_rate(tickers):
    try: usd_krw_rate = yf.Ticker("USDKRW=X").history(period='1d')['Close'].iloc[-1]
    except: usd_krw_rate = 1350.0
    try:
        data = yf.download(tickers, period='1d', progress=False)
        if data.empty: return {}, usd_krw_rate
        prices = data['Close'].iloc[-1].to_dict() if isinstance(data.columns, pd.MultiIndex) else {tickers[0]: data['Close'].iloc[-1]}
        return prices, usd_krw_rate
    except: return {}, usd_krw_rate
def create_portfolio_dashboard(df, prices, usd_krw_rate):
    dashboard_df = df.copy()
    dashboard_df['API 현재가'] = dashboard_df['종목코드'].map(prices)
    if '수동 현재가(KRW)' in dashboard_df.columns:
        dashboard_df['현재가'] = dashboard_df['수동 현재가(KRW)'].where(dashboard_df['수동 현재가(KRW)'] > 0, dashboard_df['API 현재가'])
    else: dashboard_df['현재가'] = dashboard_df['API 현재가']
    dashboard_df['통화'] = 'USD'
    dashboard_df.loc[dashboard_df['종목코드'].str.contains('.KS|.KQ', na=False), '통화'] = 'KRW'
    dashboard_df['평균 단가 (고유)'] = dashboard_df.apply(lambda r: r['평균 단가(KRW)'] if r['통화'] == 'KRW' else r['평균 단가(USD)'], axis=1)
    dashboard_df['현재가 (고유)'] = dashboard_df['현재가']
    dashboard_df['총 매수 금액 (KRW)'] = dashboard_df.apply(lambda r: (r['수량'] * r['평균 단가 (고유)']) if r['통화'] == 'KRW' else (r['수량'] * r['평균 단가 (고유)'] * usd_krw_rate), axis=1)
    dashboard_df['현재 평가 금액 (KRW)'] = pd.NA
    if '수동 수익률(%)' in dashboard_df.columns:
        mask = dashboard_df['수동 수익률(%)'] != 0
        dashboard_df.loc[mask, '현재 평가 금액 (KRW)'] = dashboard_df.loc[mask, '총 매수 금액 (KRW)'] * (1 + dashboard_df.loc[mask, '수동 수익률(%)'])
    auto_calc_mask = dashboard_df['현재 평가 금액 (KRW)'].isna()
    dashboard_df.loc[auto_calc_mask, '현재 평가 금액 (KRW)'] = dashboard_df.loc[auto_calc_mask].apply(lambda r: (r['수량'] * r['현재가']) if r['통화'] == 'KRW' else (r['수량'] * r['현재가'] * usd_krw_rate), axis=1)
    dashboard_df['손익 (KRW)'] = (dashboard_df['현재 평가 금액 (KRW)'] - dashboard_df['총 매수 금액 (KRW)']).fillna(0)
    dashboard_df['손익 (고유)'] = dashboard_df.apply(lambda r: r['손익 (KRW)'] if r['통화'] == 'KRW' else r['손익 (KRW)'] / usd_krw_rate, axis=1)
    dashboard_df['수익률 (%)'] = (dashboard_df['손익 (KRW)'] / dashboard_df['총 매수 금액 (KRW)'].replace(0, pd.NA)) * 100
    return dashboard_df
@st.cache_data
def get_peer_summary(ticker_list):
    summary_data = []
    for ticker in ticker_list:
        try:
            profile, quote = get_company_profile(ticker), get_quote(ticker)
            summary_data.append({"Ticker": ticker, "Name": profile.get('name', ticker), "Market Cap (M)": profile.get('marketCapitalization', 0), "% Change": quote.get('dp', 0)})
        except: continue
    return pd.DataFrame(summary_data)

# --- 4. 메인 UI 및 로직 ---
st.title("💎 GEM: Finance Dashboard")
st.caption("v34.0 - Final Strategy Implemented")

if 'data_loaded' not in st.session_state:
    with st.spinner("Initializing System... Loading data from Google Sheets..."):
        st.session_state.portfolio_df, st.session_state.watchlist_df, st.session_state.cash_df = load_data_from_gsheet()
    st.session_state.data_loaded = True

if 'active_view' not in st.session_state:
    st.session_state.active_view = "💼 포트폴리오"

# --- 포트폴리오 뷰 ---
if st.session_state.active_view == "💼 포트폴리오":
    st.header("💼 Portfolio Command Center")
    
    portfolio_df = st.session_state.portfolio_df
    cash_df = st.session_state.cash_df

    if not portfolio_df.empty or not cash_df.empty:
        all_portfolio_tickers = [ticker for ticker in portfolio_df['종목코드'].tolist() if ticker]
        invest_dashboard_df = pd.DataFrame()
        if all_portfolio_tickers:
            current_prices, usd_krw_rate = get_current_prices_and_rate(all_portfolio_tickers)
            st.sidebar.metric("USD/KRW 환율", f"₩{usd_krw_rate:,.2f}")
            invest_dashboard_df = create_portfolio_dashboard(portfolio_df, current_prices, usd_krw_rate)
        
        cash_dashboard_df = pd.DataFrame()
        if not cash_df.empty:
            cash_dashboard_df = cash_df.rename(columns={'금액(KRW)': '현재 평가 금액 (KRW)'})
            cash_dashboard_df['수익률 (%)'] = 0; cash_dashboard_df['손익 (고유)'] = 0; cash_dashboard_df['수량'] = '-'; cash_dashboard_df['평균 단가 (고유)'] = '-'; cash_dashboard_df['현재가 (고유)'] = '-';
        
        display_cols = ['계좌구분', '종목명', '자산티어', '수량', '평균 단가 (고유)', '현재가 (고유)', '손익 (고유)', '수익률 (%)', '현재 평가 금액 (KRW)']
        final_dashboard_df = pd.concat([
            invest_dashboard_df.reindex(columns=display_cols),
            cash_dashboard_df.reindex(columns=display_cols)
        ], ignore_index=True).fillna('-')

        total_value = pd.to_numeric(final_dashboard_df['현재 평가 금액 (KRW)'], errors='coerce').sum()
        total_cost = invest_dashboard_df['총 매수 금액 (KRW)'].sum() if not invest_dashboard_df.empty else 0
        total_pl = invest_dashboard_df['손익 (KRW)'].sum() if not invest_dashboard_df.empty else 0
        total_pl_percent = (total_pl / total_cost) * 100 if total_cost > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("총 평가 자산", f"₩{total_value:,.0f}")
        col2.metric("총 손익 (투자자산)", f"₩{total_pl:,.0f}", f"{total_pl_percent:.2f}%")
        col3.metric("총 투자 원금", f"₩{total_cost:,.0f}")
        st.divider()

        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.subheader("보유 자산 상세")
            st.dataframe(final_dashboard_df[display_cols].style.format({
                '손익 (고유)': '{:,.2f}', '수익률 (%)': '{:.2f}%', '현재 평가 금액 (KRW)': '₩{:,.0f}'
            }, na_rep="-").background_gradient(cmap='RdYlGn', subset=['수익률 (%)']), use_container_width=True)
        
        with col2:
            st.subheader("자산 배분")
            chart_group_by = st.radio("차트 기준", ['자산티어', '계좌구분'], horizontal=True, key='chart_group')
            filter_cols = st.columns(2)
            exclude_base = filter_cols[0].checkbox("'기반' 티어 제외", value=True)
            exclude_cash = filter_cols[1].checkbox("'현금' 자산 제외", value=True)
            chart_df = final_dashboard_df.copy()
            chart_df['현재 평가 금액 (KRW)'] = pd.to_numeric(chart_df['현재 평가 금액 (KRW)'], errors='coerce').fillna(0)
            if exclude_base and '자산티어' in chart_df.columns: chart_df = chart_df[~chart_df['자산티어'].str.contains('기반', na=False)]
            if exclude_cash and '자산티어' in chart_df.columns: chart_df = chart_df[~chart_df['자산티어'].str.contains('현금', na=False)]
            if not chart_df.empty and chart_df['현재 평가 금액 (KRW)'].sum() > 0:
                allocation = chart_df.groupby(chart_group_by)['현재 평가 금액 (KRW)'].sum()
                fig_tier = px.pie(values=allocation.values, names=allocation.index, title=f"{chart_group_by}별 비중", hole=.3)
                st.plotly_chart(fig_tier, use_container_width=True)
            else: st.warning("차트에 표시할 데이터가 없습니다.")
    else:
        st.info("Google Sheets에서 포트폴리오 또는 현금 데이터를 불러올 수 없습니다.")

# --- 상세 분석 뷰 ---
elif st.session_state.active_view == "🔍 상세 분석":
    if 'analysis_tickers' in st.session_state and st.session_state.analysis_tickers:
        main_ticker = st.session_state.analysis_tickers[0]
        st.header(f"🔍 {main_ticker} 상세 분석")
        
        with st.spinner(f"'{main_ticker}' 상세 데이터를 가져오는 중..."):
            profile, quote, news, financials_df, peers, earnings_data, next_earnings_date, candles_df = (get_company_profile(main_ticker), get_quote(main_ticker), get_company_news(main_ticker), get_basic_financials(main_ticker), get_company_peers(main_ticker), get_company_earnings(main_ticker), get_earnings_calendar(main_ticker), get_stock_candles(main_ticker))
        
        analysis_tab_names = ["💎 GEMINI 분석", "📜 과거 분석 기록", "📊 개요", "📈 기술적 분석", "💰 재무", "👥 경쟁사 비교", "📈 실적", "📰 뉴스"]
        gemini_tab, log_tab, overview_tab, tech_tab, fin_tab, peer_tab, earn_tab, news_tab = st.tabs(analysis_tab_names)

        with gemini_tab:
            st.subheader(f"💎 {main_ticker} AI 전략 분석")

            # 1. 분석 실행 버튼
            if st.button("🚀 GEMINI 전략 분석 실행", use_container_width=True, type="primary", key=f"gemini_run_{main_ticker}"):
                with st.spinner("GEMINI가 전략을 분석 중입니다..."):
                    # 이전 분석 상태를 모두 초기화
                    st.session_state.last_analysis_text = None
                    st.session_state.last_saved_ticker = None
                    
                    # AI에게 보낼 컨텍스트 생성
                    portfolio_df = st.session_state.portfolio_df
                    cash_df = load_cash_data_only()
                    
                    holding_context = f"현재 {main_ticker} 종목은 보유하고 있지 않습니다."
                    if not portfolio_df.empty and main_ticker in portfolio_df['종목코드'].values:
                        holding_info = portfolio_df[portfolio_df['종목코드'] == main_ticker].iloc[0]
                        shares, avg_price_usd = holding_info.get('수량', 0), holding_info.get('평균 단가(USD)', 0)
                        holding_context = f"현재 분석 대상인 {main_ticker} 종목은 {shares}주를 평균 단가 ${avg_price_usd:,.2f}에 보유 중입니다."

                    investable_cash, emergency_cash = 0, 0
                    if not cash_df.empty:
                        investable_mask = cash_df['종목명'].str.contains('CMA', na=False)
                        investable_cash = cash_df.loc[investable_mask, '금액(KRW)'].sum()
                        emergency_mask = cash_df['종목명'].str.contains('비상금', na=False)
                        emergency_cash = cash_df.loc[emergency_mask, '금액(KRW)'].sum()
                    
                    cash_context = f"그 외에, 추가로 투자 가능한 현금(CMA)은 약 {investable_cash:,.0f}원 보유 중이며, 비상금은 약 {emergency_cash:,.0f}원입니다."
                    full_context = f"{holding_context}\n{cash_context}"
                    
                    tech_summary = generate_technical_summary(add_technical_indicators(candles_df.copy()))
                    
                    # 이제 rerun 없이, 직접 스트리밍하고 결과를 session_state에 저장
                    analysis_generator = stream_and_capture_analysis(main_ticker, profile, quote, financials_df, tech_summary, news, full_context)
                    
                    # 스트리밍 결과를 화면에 표시
                    st.markdown("---")
                    st.write_stream(analysis_generator)
                    st.session_state.last_analysis_ticker = main_ticker


            # 2. [핵심 수정] 분석 결과가 존재하면 "무조건" 표시 및 저장 버튼 관리
            if st.session_state.get("last_analysis_text") and st.session_state.get("last_analysis_ticker") == main_ticker:
                # 저장 후에도 내용이 사라지지 않도록, 스트리밍이 아닌 최종 텍스트를 항상 표시
                st.markdown("---")
                
                st.divider()

                if "오류" not in st.session_state.last_analysis_text:
                    if st.session_state.get("last_saved_ticker") == main_ticker:
                        st.success("✅ 이 분석 결과는 '과거 분석 기록'에 저장되었습니다.")
                    else:
                        if st.button("💾 현재 분석 결과 저장", key=f"gemini_save_{main_ticker}"):
                            with st.spinner("분석 결과를 Google Sheets에 저장하는 중..."):
                                analysis_text = st.session_state.get("last_analysis_text", "")
                                try:
                                    summary_text = analysis_text.split("#### 4. 어떻게 행동해야 하는가? (How to Act?)")[1].strip().replace("*", "").replace("#", "")[:200] + "..."
                                except (IndexError, AttributeError):
                                    summary_text = analysis_text.strip().replace("*","").replace("#","")[:200] + "..."
                                
                                log_entry = {
                                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "종목코드": main_ticker, "AI_Model": st.session_state.get("last_model_used", "N/A"),
                                    "당시 주가": quote.get('c', 0), "분석 요약": summary_text,
                                    "전체 분석 내용": analysis_text, "주요 데이터": json.dumps({}, ensure_ascii=False)
                                }
                                if save_analysis_to_gsheet(log_entry):
                                    st.session_state.last_saved_ticker = main_ticker
                                    st.cache_data.clear()
                                    st.rerun()
                                else:
                                    st.error("분석 결과 저장에 실패했습니다.")
        
        with log_tab:
            st.subheader(f"📜 {main_ticker} 과거 분석 기록")
            analysis_logs = load_analysis_log(main_ticker)
            if not analysis_logs.empty:
                for index, row in analysis_logs.iterrows():
                    with st.expander(f"**{row['Timestamp']}** | 당시 주가: ${float(row.get('당시 주가', 0)):.2f} | 모델: {row.get('AI_Model', 'N/A')}"):
                        st.markdown(f"**요약:** {row.get('분석 요약', 'N/A')}")
                        st.markdown("---")
                        st.markdown(row['전체 분석 내용'])
            else: st.info(f"'{main_ticker}'에 대한 과거 분석 기록이 없습니다.")
        
        with overview_tab:
            if profile:
                st.subheader(f"{profile.get('name', main_ticker)} ({main_ticker})")
                col1, col2 = st.columns([1, 4]); col1.image(profile.get('logo'), width=100)
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
                st.subheader("다가오는 주요 이벤트"); st.info(f"**다음 실적 발표 예정일:** {next_earnings_date}")

        with tech_tab:
            st.subheader("주가 차트 및 기술적 지표")
            if not candles_df.empty and len(candles_df) > 60:
                candles_df_tech = add_technical_indicators(candles_df.copy())
                st.subheader("기술적 분석 요약")
                tech_summary = generate_technical_summary(candles_df_tech)
                for point in tech_summary: st.markdown(f"- {point}")
                st.divider()
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Candlestick & Moving Averages', 'MACD', 'RSI'), row_heights=[0.6, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=candles_df_tech['Date'], open=candles_df_tech['Open'], high=candles_df_tech['High'], low=candles_df_tech['Low'], close=candles_df_tech['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SMA20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SMA60'], name='SMA 60', line=dict(color='purple', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SignalLine'], name='Signal Line', line=dict(color='red', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['RSI14'], name='RSI 14', line=dict(color='royalblue', width=1)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.update_layout(height=800, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info(f"'{main_ticker}'에 대한 차트 데이터가 부족합니다.")
            
        with fin_tab:
            st.subheader("핵심 재무 지표 (연간)")
            if not financials_df.empty:
                st.dataframe(financials_df.style.format("{:,.2f}", na_rep="-"))
                for col in financials_df.columns:
                    if financials_df[col].notna().any():
                        fig = px.bar(financials_df.dropna(subset=[col]), y=col, title=f"Annual {col.capitalize()} (M)")
                        st.plotly_chart(fig, use_container_width=True)
            else: st.warning(f"'{main_ticker}'에 대한 재무 데이터를 찾을 수 없습니다.")

        with peer_tab:
            st.subheader("경쟁사 비교")
            if peers:
                peer_df = get_peer_summary([p for p in peers if p != main_ticker][:5])
                if not peer_df.empty:
                    st.dataframe(peer_df.set_index('Ticker').style.format({"Market Cap (M)": "{:,.0f}", "% Change": "{:.2f}%"}, na_rep="-").background_gradient(cmap='RdYlGn', subset=['% Change']))
                else: st.info("경쟁사 정보를 가져올 수 없습니다.")
            else: st.info("경쟁사 정보가 없습니다.")

        with earn_tab:
            st.subheader("분기별 실적 발표 내역")
            if not earnings_data.empty:
                format_dict = {'실제 EPS': '{:.2f}', '예상 EPS': '{:.2f}', 'EPS 서프라이즈 (%)': '{:.2f}%'}
                st.dataframe(earnings_data.style.format(format_dict, na_rep="-"))
                if 'EPS 서프라이즈 (%)' in earnings_data.columns:
                    fig = px.bar(earnings_data, x='발표 분기', y='EPS 서프라이즈 (%)', color='EPS 결과', color_discrete_map={'Beat': 'green', 'Miss': 'red', 'Meet': 'blue'})
                    st.plotly_chart(fig, use_container_width=True)
            else: st.warning(f"'{main_ticker}'에 대한 실적 내역이 없습니다.")

        with news_tab:
            st.subheader("최신 관련 뉴스")
            if news:
                for item in news[:10]:
                    news_date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M')
                    st.markdown(f"**[{item['headline']}]({item['url']})**\n- *Source: {item['source']} | {news_date}*")
            else: st.info("관련 뉴스가 없습니다.")
    else:
        st.info("사이드바에서 분석할 Ticker를 입력하고 '분석 실행' 버튼을 클릭하여 상세 분석을 시작하세요.")


        
with st.sidebar:
    st.header("Controls")
    view_options = ["💼 포트폴리오", "🔍 상세 분석"]
    active_view_index = view_options.index(st.session_state.active_view)
    st.session_state.active_view = st.radio("Select View", view_options, index=active_view_index, horizontal=True)
    st.divider()
    
    default_tickers = ""
    if not st.session_state.watchlist_df.empty and '종목코드' in st.session_state.watchlist_df.columns:
        default_tickers = ", ".join(st.session_state.watchlist_df['종목코드'].dropna().unique().tolist())
    
    tickers_input = st.text_area("Ticker(s) for Analysis", value=default_tickers, help="분석할 종목의 Ticker를 쉼표(,)로 구분하여 입력하세요.")
    
    if st.button("🔍 분석 실행", use_container_width=True, type="primary"):
        st.session_state.analysis_tickers = [ticker.strip().upper() for ticker in tickers_input.replace(',', '\n').split('\n') if ticker.strip()]
        st.session_state.active_view = "🔍 상세 분석"
        st.session_state.last_analysis_text = None
        st.session_state.last_saved_ticker = None
        st.rerun()

    st.divider()
    st.info("포트폴리오, 현금, 관심종목은 Google Sheets에서 직접 수정해주세요.")
    if st.button("🔄 Reload Data & Clear Cache", use_container_width=True):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()
