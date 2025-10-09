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
import re  # Add this line
from googleapiclient.discovery import build # <-- 이 라인을 추가
from io import BytesIO
from googleapiclient.http import MediaIoBaseDownload
import tempfile # <-- 이 라인을 추가
import os       # <-- 이 라인을 추가
from PyPDF2 import PdfReader
from fredapi import Fred # 파일 상단 import 부분에 추가


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

    # [신규] GEM-Core AI 모델을 중앙에서 관리하는 함수
    @st.cache_resource
    def get_gem_core_ai():
        print("Initializing GEM-Core AI Model...") # AI가 처음 로드될 때만 이 메시지가 표시됩니다.
        model_name = 'gemini-2.5-pro' # 사용할 모델 지정
        return genai.GenerativeModel(model_name)


except Exception as e:
    st.error(f"API 키 또는 인증 정보를 secrets.toml 파일에 설정해주세요. 오류: {e}")
    st.stop()

# [최종 단순화 버전] run_status_screener 함수를 아래 코드로 교체
def run_status_screener(ticker_list):
    """
    (최종 단순화 버전) 전체 종목에 대해 '매수', '주의/매도', '중립'의 3가지 상태로만 판정합니다.
    """
    if not ticker_list: return pd.DataFrame()

    data = yf.download(ticker_list, period="1y", progress=False, auto_adjust=True)
    if data.empty: return pd.DataFrame()

    final_results = []
    progress_bar = st.progress(0, text="전체 종목 상태 분석 시작...")

    for i, ticker in enumerate(ticker_list):
        progress_bar.progress((i + 1) / len(ticker_list), text=f"분석 중... {ticker}")
        try:
            stock_df = data.loc[:, (slice(None), ticker)]
            stock_df.columns = stock_df.columns.droplevel(1)
            
            if stock_df.empty or len(stock_df) < 61: continue

            # 지표 계산 (SMA, 52주 고점, RSI)
            sma20 = stock_df['Close'].rolling(window=20).mean()
            sma60 = stock_df['Close'].rolling(window=60).mean()
            high_52w = stock_df['High'].rolling(window=252, min_periods=1).max().iloc[-1]
            mdd_percent = (stock_df['Close'].iloc[-1] / high_52w - 1) * 100 if high_52w > 0 else 0
            
            delta = stock_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
            rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
            latest_rsi = rsi.iloc[-1]

            # --- ✨ 상태 판정 로직 단순화 ---
            status = "중립/관망 ⚪️"
            # 매수 신호
            if (sma20.iloc[-2] < sma60.iloc[-2] and sma20.iloc[-1] > sma60.iloc[-1]) or (latest_rsi <= 30):
                status = "매수 신호 🟢"
            # 주의/매도 신호
            elif (sma20.iloc[-2] > sma60.iloc[-2] and sma20.iloc[-1] < sma60.iloc[-1]) or (latest_rsi >= 70):
                status = "주의/매도 신호 🔴"

            final_results.append({
                "종목코드": ticker,
                "상태": status,
                "현재가": stock_df['Close'].iloc[-1],
                "등락률(%)": (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-2] - 1) * 100,
                "고점대비(%)": mdd_percent,
                "RSI": latest_rsi
            })
        except Exception:
            continue
    
    progress_bar.empty()
    return pd.DataFrame(final_results)



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

# [최종 버전] 모든 기능을 통합한 단일 동기화 함수
def synchronize_knowledge_files(folder_name="GEM_Finance_Knowledge", core_folder_name="Core_Principles"):
    """Google Drive 폴더와 그 하위 폴더를 모두 탐색하여 Gemini API와 동기화하고,
    '최상위 지침'과 '참고 자료' 파일 목록을 구분하여 반환합니다."""
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        
        # 1. 메인 폴더 ID 찾기
        folder_response = drive_service.files().list(q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false", fields='files(id)').execute()
        if not folder_response.get('files'):
            st.warning(f"Google Drive에서 '{folder_name}' 폴더를 찾을 수 없습니다.")
            return [], []
        folder_id = folder_response.get('files')[0].get('id')
        
        # 2. [핵심 수정] 메인 폴더 하위의 모든 파일과 폴더를 한 번에 가져오기
        all_items_response = drive_service.files().list(q=f"'{folder_id}' in parents and trashed=false", fields='files(id, name, mimeType, modifiedTime, parents)').execute()
        all_items = all_items_response.get('files', [])

        # Core_Principles 폴더 ID 찾기 및 해당 폴더 내부 파일 가져오기
        core_folder_id = None
        for item in all_items:
            if item.get('mimeType') == 'application/vnd.google-apps.folder' and item.get('name') == core_folder_name:
                core_folder_id = item.get('id')
                core_files_response = drive_service.files().list(q=f"'{core_folder_id}' in parents and trashed=false", fields='files(id, name, mimeType, modifiedTime, parents)').execute()
                all_items.extend(core_files_response.get('files', []))
                break
        
        # 파일만 필터링하고, Core_Principles 파일 이름 목록 생성
        drive_files = {f['name']: f for f in all_items if f.get('mimeType') != 'application/vnd.google-apps.folder'}
        core_file_names = {f['name'] for f in all_items if f.get('parents') and core_folder_id in f.get('parents')}

        # 3. Gemini API와 동기화
        gemini_files_list = genai.list_files()
        gemini_files = {f.display_name: f for f in gemini_files_list}

        # 3. Drive 기준으로 동기화
        with st.spinner("Knowledge Core 동기화 중..."):
            # Drive에 없는 파일은 Gemini에서 삭제
            for name, gemini_file in gemini_files.items():
                if name not in drive_files:
                    st.write(f"   - Drive에서 삭제된 파일 '{name}'을 AI에서 제거합니다.")
                    genai.delete_file(gemini_file.name)
            
            # Drive에 새로 추가/수정된 파일은 Gemini에 업로드
            for name, drive_file in drive_files.items():
                should_upload = False
                if name in gemini_files:
                    gemini_file_metadata = genai.get_file(gemini_files[name].name)
                    drive_mod_time = pd.to_datetime(drive_file['modifiedTime'])
                    gemini_create_time = pd.to_datetime(gemini_file_metadata.create_time)
                    if drive_mod_time > gemini_create_time:
                        st.write(f"   - 수정된 파일 '{name}'을 AI에 다시 업로드합니다.")
                        genai.delete_file(gemini_files[name].name)
                        should_upload = True
                else:
                    should_upload = True

                if should_upload:
                    st.write(f"   - 새 파일 '{name}'을 AI에 업로드합니다.")
                    request = drive_service.files().get_media(fileId=drive_file.get('id'))
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{name}") as tmp_file:
                        fh = BytesIO()
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()
                        tmp_file.write(fh.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    genai.upload_file(path=tmp_file_path, display_name=name)
                    os.unlink(tmp_file_path)

        st.success("Knowledge Core 동기화 완료!")

       # 4. 최종 파일 목록을 구분하여 반환
        final_gemini_files = genai.list_files()
        core_principle_files = [f for f in final_gemini_files if f.display_name in core_file_names]
        #reference_files = [f for f in final_gemini_files if f.display_name not in core_file_names]
        
        return core_principle_files#, reference_files

    except Exception as e:
        st.error(f"지식 파일 동기화 중 오류: {e}")
        return [], []


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

# [신규] 중앙 데이터 허브를 경유하여 시세 정보를 가져오는 함수
def get_quote_from_hub(ticker):
    """
    중앙 데이터 허브(st.session_state.data_hub)를 확인하고,
    데이터가 없거나 5분이 지났으면 API를 호출하여 갱신합니다.
    """
    now = datetime.now()
    hub_key = f"quote_{ticker}"
    
    # 허브에 데이터가 있고, 5분 이내의 최신 정보인지 확인
    if hub_key in st.session_state.data_hub:
        data, timestamp = st.session_state.data_hub[hub_key]
        if (now - timestamp) < timedelta(minutes=5):
            return data # 최신 정보가 있으면 바로 반환

    # 허브에 없거나 오래된 정보이면, 실제 API 호출 (기존 함수 재활용)
    # st.write(f"CACHE MISS: Calling API for {ticker} quote...") # 테스트용 로그
    new_data = get_quote(ticker)
    
    # 허브에 최신 정보와 타임스탬프 저장
    if new_data:
        st.session_state.data_hub[hub_key] = (new_data, now)
        
    return new_data


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

@st.cache_data(ttl=3600) # 1시간마다 Google Sheets에서 티커 목록 갱신
def get_all_ticker_lists():
    """'Tickers' 시트에서 모든 종목 리스트를 읽어와 딕셔너리 형태로 반환합니다."""
    try:
        tickers_ws = gc.open(SPREADSHEET_NAME).worksheet("Tickers")
        all_values = tickers_ws.get_all_values()
        if not all_values:
            return {}
        
        headers = all_values[0]
        ticker_lists = {header: [] for header in headers}
        
        for col_idx, header in enumerate(headers):
            for row_idx in range(1, len(all_values)):
                if col_idx < len(all_values[row_idx]) and all_values[row_idx][col_idx]:
                    ticker_lists[header].append(all_values[row_idx][col_idx])
        return ticker_lists
    except Exception as e:
        st.error(f"Google Sheets 'Tickers' 시트를 불러오는 데 실패했습니다: {e}")
        return {}




def add_technical_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    # [수정] 누락되었던 120일, 200일 이동평균선 계산 추가
    df['SMA120'] = df['Close'].rolling(window=120).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
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

def calculate_support_levels(df):
    """주가 데이터프레임을 받아 AI가 참고할 잠재적 지지선 리스트를 반환합니다."""
    if df.empty or len(df) < 200:
        return []
    high_1y = df['High'].max()
    low_1y = df['Low'].min()
    
    fibo_382 = high_1y - (high_1y - low_1y) * 0.382
    fibo_500 = high_1y - (high_1y - low_1y) * 0.5
    fibo_618 = high_1y - (high_1y - low_1y) * 0.618
    
    # SMA 120, 200일선 계산을 위해 add_technical_indicators 함수 재사용
    df_with_sma = add_technical_indicators(df.copy())
    sma_120 = df_with_sma['SMA120'].iloc[-1]
    sma_200 = df_with_sma['SMA200'].iloc[-1]

    supports = [fibo_382, fibo_500, fibo_618, sma_120, sma_200]
    return [f"${s:.2f}" for s in supports if pd.notna(s)]

# [최종 수정] 계층적 지식 시스템을 명시적 'Tool'로 구현한 최종 AI 분석 함수
def stream_and_capture_analysis(ticker, profile, quote, financials_df, tech_summary, news, portfolio_context, support_levels, dynamic_trends, market_context, core_principle_files):
    model = get_gem_core_ai()
    model_name = model.model_name
    
    # [핵심 수정] 모델 생성 시 tools 선언을 제거
    model = genai.GenerativeModel(model_name)

    # 동적 트렌드 설명을 위한 텍스트 생성
    trends_text = f"""
       - 실시간: {dynamic_trends.get('realtime_change_percent', 0.0):.2f}%
       - 3일 수익률: {dynamic_trends.get('return_3d_percent', 'N/A')}
       - 7일 수익률: {dynamic_trends.get('return_7d_percent', 'N/A')}
       - 30일 수익률: {dynamic_trends.get('return_30d_percent', 'N/A')}
       - 30일 S&P500 대비: {dynamic_trends.get('vs_spy_30d_percent', 'N/A')}
    """
    
    # [추가] 거시 경제 컨텍스트를 위한 텍스트 생성
    market_context_text = f"""
       - VIX (공포지수): {market_context.get('VIX', {}).get('price', 'N/A'):.2f} (20 이상일 경우 변동성 확대 주의)
       - 미국 10년물 국채금리: {market_context.get('US 10Y', {}).get('price', 'N/A'):.2f}% (금리 상승은 일반적으로 성장주에 부담)
    """
# 프롬프트 내에서 각 파일의 역할을 명확히 구분
    core_files_names = ", ".join([f.display_name for f in core_principle_files]) if core_principle_files else "없음"
    #ref_files_names = ", ".join([f.display_name for f in reference_files]) if reference_files else "없음"
    
    master_prompt = f"""
    **SYSTEM ROLE:** 당신은 월스트리트 최고의 금융 분석가이자, 'MASTER'라는 투자자를 보좌하는 AI 전략 파트너, 'GEM: Finance'입니다. 당신의 분석은 항상 MASTER의 투자 철학이 담긴 최상위 지침을 최우선으로 합니다.

    **INPUT DATA:**
    - 분석 대상: {ticker}
    - MASTER 포트폴리오 상황: {portfolio_context}
    - 기업 개요: {profile.get('name', 'N/A')}, 산업: {profile.get('finnhubIndustry', 'N/A')}
    - 현재 시세: ${quote.get('c', 0):.2f}
    - 핵심 재무 요약: \n{financials_df.tail(3).to_string() if not financials_df.empty else "N/A"}
    - 기술적 분석 요약 (시스템): \n- {"\n- ".join(tech_summary)}
    - 최신 뉴스 요약 (시스템): \n- {"\n- ".join([item['headline'] for item in news[:5]]) if news else "N/A"}
    - 잠재적 지지선 (시스템): {support_levels}
    - 최근 주가 동향: {trends_text}
    - 현재 거시 경제 상황: {market_context_text}
    - 첨부된 최상위 지침 파일: {core_files_names}
    

    **MISSION:**
    모든 INPUT DATA와 첨부된 지침 파일을 종합하여, 아래 4가지 핵심 질문에 대한 '상세한 서술형' 답변이 포함된 종합 분석 보고서를 생성하십시오.
    답변을 생성할 때, 각 질문에 해당하는 내용 안에 아래에 명시된 **7개의 데이터 블록**을 반드시 자연스럽게 포함시켜야 합니다. 모든 내용은 한국어로 작성하십시오.

    ---
    ### 💎 {ticker} 종합 분석 보고서
    *Analysis Model: `{model_name}`*

    #### 1. 좋은 종목인가? (펀더멘털 및 뉴스 분석)
    *이곳에 자유롭게 서술형으로 분석을 시작합니다...*
    [FUNDAMENTAL_ANALYSIS_BRIEFING]
    (서술 내용 요약) 제공된 재무제표의 성장성, 수익성, 안정성 동향을 분석하고, 투자자가 유의해야 할 긍정적/부정적 포인트를 요약합니다.
    [/FUNDAMENTAL_ANALYSIS_BRIEFING]
    *이어서 계속 자유롭게 서술합니다...*
    [NEWS_ANALYSIS_BRIEFING]
    (서술 내용 요약) 제공된 최신 뉴스 헤드라인들을 한국어로 요약하고, 전체적인 뉴스 흐름이 주가에 미칠 영향을 긍정적, 부정적, 중립적으로 판단하여 분석합니다.
    [/NEWS_ANALYSIS_BRIEFING]
    *결론적으로 이 종목은 펀더멘털과 뉴스 관점에서...*

    #### 2. 좋은 시기인가? (기술적 분석)
    *이곳에 자유롭게 서술형으로 분석을 시작합니다...*
    [TECHNICAL_ANALYSIS_BRIEFING]
    (서술 내용 요약) 현재 주가 차트의 주요 이평선(SMA), RSI, MACD 지표와 최근 캔들 패턴을 종합적으로 해석하여, 현재 기술적 상태가 강세인지, 약세인지, 혹은 횡보 상태인지를 분석합니다.
    [/TECHNICAL_ANALYSIS_BRIEFING]
    *따라서 현재 기술적 관점에서는...*

    #### 3. 좋은 가격인가? (매수 및 매도 전략)
    *이곳에 자유롭게 서술형으로 분석을 시작합니다...*
    [BUY_ZONES]
    zone1_start: [숫자 또는 N/A]
    zone1_end: [숫자 또는 N/A]
    zone2_start: [숫자 또는 N/A]
    zone2_end: [숫자 또는 N/A]
    rationale: 해당 매수 구간을 설정한 기술적, 펀더멘털적 근거를 간략히 서술합니다.
    [/BUY_ZONES]
    *또한, 수익 실현 관점에서는...*
    [SELL_ZONES]
    zone1_start: [숫자 또는 N/A]
    zone1_end: [숫자 또는 N/A]
    zone2_start: [숫자 또는 N/A]
    zone2_end: [숫자 또는 N/A]
    rationale: 해당 매도/수익실현 구간을 설정한 기술적, 펀더멘털적 근거를 간략히 서술합니다.
    [/SELL_ZONES]

    #### 4. 최종적으로 어떻게 행동해야 하는가? (최종 권고)
    *MASTER의 투자 철학과 모든 분석을 종합했을 때, 최종적인 행동 권고는 다음과 같습니다...*
    [RECOMMENDATION]
    action: [매수 추천, 적극 매수, 관망, 비중 축소, 매도 고려] 중 하나를 선택
    rationale: 위 투자 행동을 결정한 핵심적인 이유를 2~3문장으로 요약합니다.
    [/RECOMMENDATION]
    """

    full_response = []
    try:
        # [핵심 수정] generate_content 호출 시, 프롬프트와 파일 객체 리스트를 함께 전달
        all_files = core_principle_files#+ reference_files
        
        response = model.generate_content([master_prompt] + all_files, stream=True)

        for chunk in response:
            full_response.append(chunk.text)
            yield chunk.text

        # [수정] 대화 기록에 현재 문답 추가
        final_text = "".join(full_response)
        st.session_state.last_analysis_text = final_text
        st.session_state.last_model_used = model_name
        


    except Exception as e:
        error_message = f"Gemini 분석 중 오류가 발생했습니다: {e}"
        st.session_state.last_analysis_text = error_message
        st.session_state.last_model_used = "Error"
        yield error_message

# [✨ NEW] 1. '진화된 보고서' 생성을 위한 AI 스트리밍 함수
def stream_evolved_report(previous_analysis_text, change_summary, ticker):
    """
    기존 분석과 변경점을 바탕으로 '진화된 전체 보고서'를 생성하는 AI 함수.
    """
    if "유의미한 데이터 변경점이 감지되지 않았습니다" in change_summary:
        yield previous_analysis_text # 변경 없으면 이전 보고서 그대로 반환
        return

    model = get_gem_core_ai()
    
    prompt = f"""
**MISSION:** 아래 [기존 분석 보고서]를 [최신 데이터 변경점 요약]을 반영하여, 논리적 흐름과 구조를 완벽하게 유지한 '완결된 최신 버전의 전체 보고서'로 업데이트 하십시오. 최종 결과물은 보고서 전문이어야 합니다.

**[기존 분석 보고서]**
---
{previous_analysis_text}
---

**[최신 데이터 변경점 요약]**
---
{change_summary}
---
"""
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        yield chunk.text

# [✨ NEW] 2. '변경 요약 브리핑(Changelog)' 생성을 위한 AI 함수
@st.cache_data(ttl=3600) # 1시간 캐싱으로 중복 호출 방지
def generate_changelog(previous_analysis_text, new_analysis_text, change_summary):
    """
    이전/신규 보고서와 변경점을 비교하여 'AI 변경점 브리핑'을 생성하는 AI 함수.
    """
    if "유의미한 데이터 변경점이 감지되지 않았습니다" in change_summary:
        return "✅ 유의미한 데이터 변경점이 감지되지 않아 기존 분석이 그대로 유지됩니다."
    
    model = get_gem_core_ai()
    
    prompt = f"""
**MISSION:** 당신은 분석팀장입니다. 아래 세 가지 정보를 바탕으로, 어떤 '데이터 변경점' 때문에 이전 보고서가 새로운 보고서로 어떻게 바뀌었는지 핵심적인 이유를 '변경 요약 브리핑' 형식으로 보고하십시오.

**[1. 데이터 변경점]**
{change_summary}

**[2. 이전 보고서]**
{previous_analysis_text}

**[3. 새로운 보고서]**
{new_analysis_text}

**보고 형식:**
💡 **AI 변경점 브리핑 (Changelog)**
* **[사유]** (데이터 변경점 요약)
    * **[결과]** (보고서의 어떤 부분이 어떻게 수정되었는지 설명)
"""
    
    response = model.generate_content(prompt)
    return response.text


def structure_recommendation(full_analysis_text):
    """
    AI가 생성한 '종합 분석 보고서' 텍스트에서 모든 데이터 블록을 추출하고 파싱합니다.
    """
    if not full_analysis_text:
        return {}

    def extract_block(block_name, text):
        """지정된 이름의 데이터 블록 내용을 추출하는 도우미 함수"""
        pattern = re.compile(f'\\[{block_name}\\](.*?)\\[/{block_name}\\]', re.DOTALL)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        return None

    def parse_zones(zone_content):
        """BUY_ZONES 또는 SELL_ZONES의 내용을 파싱하는 도우미 함수"""
        if not zone_content:
            return None

        reco = {}
        for line in zone_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                reco[key.strip()] = value.strip()

        def to_float_or_none(val):
            try: return float(val)
            except (ValueError, TypeError): return None

        zone1_start = to_float_or_none(reco.get('zone1_start'))
        zone1_end = to_float_or_none(reco.get('zone1_end'))
        zone2_start = to_float_or_none(reco.get('zone2_start'))
        zone2_end = to_float_or_none(reco.get('zone2_end'))

        return {
            'zone1': (zone1_start, zone1_end) if zone1_start and zone1_end else None,
            'zone2': (zone2_start, zone2_end) if zone2_start and zone2_end else None,
            'rationale': reco.get('rationale', 'N/A')
        }

    def parse_recommendation(reco_content):
        """RECOMMENDATION 블록을 파싱하는 도우미 함수"""
        if not reco_content:
            return None

        reco = {}
        for line in reco_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                reco[key.strip()] = value.strip()
        return {
            'action': reco.get('action', 'N/A'),
            'rationale': reco.get('rationale', 'N/A')
        }

    # 각 블록의 내용을 추출
    recommendation_content = parse_recommendation(extract_block('RECOMMENDATION', full_analysis_text))
    fundamental_content = extract_block('FUNDAMENTAL_ANALYSIS_BRIEFING', full_analysis_text)
    news_content = extract_block('NEWS_ANALYSIS_BRIEFING', full_analysis_text)
    technical_content = extract_block('TECHNICAL_ANALYSIS_BRIEFING', full_analysis_text)
    buy_zones_content = parse_zones(extract_block('BUY_ZONES', full_analysis_text))
    sell_zones_content = parse_zones(extract_block('SELL_ZONES', full_analysis_text))

    # 최종 구조화된 데이터로 종합
    structured_data = {
        'recommendation': recommendation_content,
        'fundamental_briefing': fundamental_content,
        'news_briefing': news_content,
        'technical_briefing': technical_content,
        'buy_zones': buy_zones_content,
        'sell_zones': sell_zones_content
    }

    return structured_data

def calculate_dynamic_trends(df, quote):
    """주가 데이터프레임과 실시간 시세를 받아 동적 트렌드 딕셔너리를 반환합니다."""
    trends = {}
    if df.empty or len(df) < 31:
        return trends

    # 실시간 등락률
    trends['realtime_change_percent'] = quote.get('dp', 0.0)

    # 기간별 수익률 계산
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    latest_price = df['Close'].iloc[-1]
    for days in [3, 7, 30]:
        try:
            past_price = df['Close'].iloc[-days-1]
            trends[f'return_{days}d_percent'] = ((latest_price - past_price) / past_price) * 100
        except IndexError:
            trends[f'return_{days}d_percent'] = 'N/A'

    # S&P500 대비 성과 (30일 기준)
    try:
        spy_df = yf.download('SPY', start=df.index[-31], end=df.index[-1] + pd.Timedelta(days=1), progress=False)
        spy_return = (spy_df['Close'].iloc[-1] - spy_df['Close'].iloc[0]) / spy_df['Close'].iloc[0] * 100
        stock_return_30d = trends.get('return_30d_percent', 0)
        if stock_return_30d != 'N/A':
            trends['vs_spy_30d_percent'] = stock_return_30d - spy_return
    except Exception:
        trends['vs_spy_30d_percent'] = 'N/A'
        
    return trends


# backup (lines 260-281)
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
            # [수정] get_quote를 get_quote_from_hub로 변경
            profile, quote = get_profile_from_hub(ticker), get_quote_from_hub(ticker) # <-- 여기를 수정
            summary_data.append({"Ticker": ticker, "Name": profile.get('name', ticker), "Market Cap (M)": profile.get('marketCapitalization', 0), "% Change": quote.get('dp', 0)})
        except: continue
    return pd.DataFrame(summary_data)


# --- [MOD v39.1 Start] 데이터 수집 로직 안정성 복원 및 개선 (최종) ---
@st.cache_data(ttl=300)
def get_market_status_data():
    """[최종 안정화 버전] 각 지표의 '최종 업데이트 날짜'를 함께 반환합니다."""
    data = {}
    tickers = {
        "S&P 500": "^GSPC", "Nasdaq": "^IXIC", "KOSPI": "^KS11", "VIX": "^VIX", 
        "US 10Y": "^TNX", "Dollar": "DX-Y.NYB", "Crude Oil": "CL=F", 
        "Gold": "GC=F", "USD/KRW": "USDKRW=X"
    }
    
    hist_data = pd.DataFrame() 
    try:
        # 1. 모든 티커 데이터를 한 번에 효율적으로 다운로드
        hist_data = yf.download(list(tickers.values()), period="6d", progress=False)
        
        if not hist_data.empty:
            for name, ticker_symbol in tickers.items():
                try:
                    # 2. 각 티커별로 데이터 시리즈를 추출하고, 비어있는 값(NaN) 제거
                    ticker_series = hist_data['Close'][ticker_symbol].dropna()
                    
                    if ticker_series.empty:
                        data[name] = {"price": "N/A", "change": "N/A", "change_percent": "N/A", "last_update": "N/A"}
                        continue

                    # 3. 데이터의 마지막 날짜를 'last_update'로 추출
                    last_update_date = ticker_series.index[-1].strftime('%Y-%m-%d')
                    
                    # 4. 등락률 계산
                    if len(ticker_series) >= 2:
                        price = ticker_series.iloc[-1]
                        change = price - ticker_series.iloc[-2]
                        change_percent = (change / ticker_series.iloc[-2]) * 100 if ticker_series.iloc[-2] != 0 else 0
                        data[name] = {"price": price, "change": change, "change_percent": change_percent, "last_update": last_update_date}
                    elif len(ticker_series) == 1:
                        data[name] = {"price": ticker_series.iloc[-1], "change": "N/A", "change_percent": "N/A", "last_update": last_update_date}
                except (KeyError, IndexError):
                    data[name] = {"price": "N/A", "change": "N/A", "change_percent": "N/A", "last_update": "N/A"}
    except Exception:
        for name in tickers.keys():
            data[name] = {"price": "N/A", "change": "N/A", "change_percent": "N/A", "last_update": "N/A"}

    try:
        data['news'] = finnhub_client.general_news('general', min_id=0)[:5]
    except Exception:
        data['news'] = []
        
    return data, hist_data
# --- [MOD v39.1 End] ---


# --- [MOD v38 Start] 테마 ETF에 '추세 분석' 기능 추가 ---
@st.cache_data(ttl=300)
def get_theme_etf_performance():
    """선정된 12개 ETF의 '당일 성과'와 '5일 추세'를 함께 반환합니다."""
    etf_tickers = {
        "S&P 500": "SPY", "가치주": "VTV", "성장주": "VUG", "반도체": "SOXX", 
        "AI": "AIQ", "로보틱스": "BOTZ", "바이오테크": "IBB", "차세대 전력": "GRID",
        "고배당": "SCHD", "장기채": "TLT", "혁신기술": "ARKK", "비트코인": "IBIT"
    }
    try:
        # 5일 추세 계산을 위해 6일치 데이터 요청
        data = yf.download(list(etf_tickers.values()), period="6d", progress=False)
        if data.empty: return {}
        
        performance = {}
        for theme, ticker in etf_tickers.items():
            series = data['Close'][ticker].dropna()
            if len(series) >= 2:
                # 당일 등락률
                change_percent = (series.iloc[-1] / series.iloc[-2] - 1) * 100
                
                # 5일 추세
                trend = "횡보"
                if len(series) >= 6: # 5일 추세를 보려면 최소 6일 데이터 필요
                    if series.iloc[-1] > series.iloc[-2] * 1.01: trend = "상승"
                    elif series.iloc[-1] < series.iloc[-2] * 0.99: trend = "하락"

                performance[theme] = {'change_percent': change_percent, 'trend': trend}
        return performance
    except Exception:
        return {}
# --- [MOD v38 End] ---


# --- [MOD v41 Start] 이벤트 캘린더 데이터 수집 함수 ---
@st.cache_data(ttl=3600) # 1시간마다 갱신
def get_economic_calendar():
    """Finnhub API를 사용하여 향후 2주간의 주요 거시 경제 이벤트를 가져옵니다."""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        two_weeks_later = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
        calendar = finnhub_client.economic_calendar(_from=today, to=two_weeks_later)
        if calendar and calendar.get('economicCalendar'):
            us_events = [
                event for event in calendar['economicCalendar'] 
                if event.get('country') == 'US' and event.get('impact') in ['high', 'medium']
            ]
            return sorted(us_events, key=lambda x: x['time'])
    except Exception:
        return []
    return []

@st.cache_data(ttl=3600) # 1시간마다 갱신
def get_portfolio_earnings_calendar(tickers):
    """입력된 티커 리스트에 대해 향후 2주간의 실적 발표일을 가져옵니다."""
    all_earnings = []
    today = datetime.now().strftime('%Y-%m-%d')
    two_weeks_later = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
    try:
        # Finnhub API는 한 번에 여러 티커 조회를 지원하지 않으므로 개별 호출
        for ticker in tickers:
            calendar = finnhub_client.earnings_calendar(
                _from=today, to=two_weeks_later, symbol=ticker
            )
            if calendar and calendar.get('earningsCalendar'):
                for earning in calendar['earningsCalendar']:
                    earning['ticker'] = ticker # 결과에 티커 정보 추가
                    all_earnings.append(earning)
        return sorted(all_earnings, key=lambda x: x['date'])
    except Exception:
        return []
# --- [MOD v41 End] ---



# --- [MOD v38.1 Start] AI 프롬프트 데이터 참조 오류 수정 및 구조 개선 (최종) ---
def generate_market_health_briefing(market_data, full_hist_data, sector_perf_df, theme_perf, combined_events, eco_indicators): # buffett_data 인수 제거


    """
    [최종 수정본] 모든 데이터를 AI 프롬프트에 '직접' 주입하고,
    '절대 규칙'을 부여하여 데이터 기반의 'JSON' 응답을 생성합니다.
    """
    model = get_gem_core_ai()
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

    # --- 데이터 요약 부분 (이전과 동일) ---
    tickers = {
        "S&P 500": "^GSPC", "Nasdaq": "^IXIC", "KOSPI": "^KS11", "VIX": "^VIX", 
        "US 10Y": "^TNX", "Dollar": "DX-Y.NYB", "Crude Oil": "CL=F", 
        "Gold": "GC=F", "USD/KRW": "USDKRW=X"
    }
    data_summary_items, news_summary_items, sector_summary_items, theme_summary_items = [], [], [], []
    
    for name, values in market_data.items():
        if name == 'news': continue
        price, change_percent = values.get('price', 'N/A'), values.get('change_percent', 'N/A')
        trend = "횡보"
        try:
            ticker_symbol = tickers.get(name)
            if ticker_symbol and not full_hist_data.empty:
                series = full_hist_data['Close'][ticker_symbol].dropna()
                if len(series) >= 5:
                    if series.iloc[-1] > series.iloc[0] * 1.01: trend = "상승 추세"
                    elif series.iloc[-1] < series.iloc[0] * 0.99: trend = "하락 추세"
        except (KeyError, IndexError, TypeError): trend = "판단 불가"
        last_update = values.get('last_update', 'N/A')
        if isinstance(price, (int, float)) and isinstance(change_percent, (int, float)):
            data_summary_items.append(f"- {name}: {price:.2f} ({change_percent:+.2f}%) | 5일 추세: {trend} | 최종 업데이트: {last_update}")
    
    news_summary_items = [f"- {news['headline']}" for news in market_data.get('news', [])]
    sector_summary_items = [f"- {row['Sector']}: {row['Performance_5D']:.2f}%" for index, row in sector_perf_df.iterrows()]
    
    for theme, perf_data in theme_perf.items():
        change = perf_data.get('change_percent', 0)
        trend = perf_data.get('trend', 'N/A')
        theme_summary_items.append(f"- {theme}: {change:+.2f}% | 5일 추세: {trend}")

    data_summary_text = "\n".join(data_summary_items) if data_summary_items else "데이터 없음"
    news_summary_text = "\n".join(news_summary_items) if news_summary_items else "최신 주요 뉴스 없음"
    sector_summary_text = "\n".join(sector_summary_items) if sector_summary_items else "섹터 데이터 없음"
    theme_summary_text = "\n".join(theme_summary_items) if theme_summary_items else "테마 ETF 데이터 없음"

    # [추가] 이벤트 데이터 요약
    event_summary_items = []
    for event in combined_events[:5]: # 상위 5개 이벤트만 요약
        event_date = datetime.strptime(event['date'], '%Y-%m-%d').strftime('%m/%d')
        if event['type'] == 'eco':
            event_summary_items.append(f"- {event_date}: {event['data']['event']}")
        elif event['type'] == 'earn':
            event_summary_items.append(f"- {event_date}: ${event['data']['ticker']} 실적발표")
    event_summary_text = "\n".join(event_summary_items) if event_summary_items else "향후 2주 내 주요 이벤트 없음"


    # [고도화] 경제 지표 요약 시, 시계열 데이터를 문자열로 변환하여 추가
    eco_summary_items = []
    if eco_indicators:
        for name, data in eco_indicators.items():
            # AI에게 12개월 시계열 데이터를 명확하게 전달
            timeseries_str = ", ".join(map(str, data.get('timeseries', [])))
            eco_summary_items.append(f"- {name} ({data['type']}): {data['value']:.2f} | 상태: {data['status']} | 12개월 추이: [{timeseries_str}]")
    eco_summary_text = "\n".join(eco_summary_items) if eco_summary_items else "데이터 없음"
    
    # --- 데이터 요약 끝 ---
    
    # ✨ --- [핵심 수정] 프롬프트 내부의 예시 문구 및 구조 개선 --- ✨
    prompt = f"""
    **SYSTEM ROLE:** 당신은 월스트리트의 수석 시장 전략가 'GEM: Finance'다. 당신의 임무는 데이터를 기반으로 객관적인 진실만을 보고하는 것이다.

    ---
    **## ✨ 절대 규칙 (Absolute Rules) ##**
    1.  **데이터 절대주의:** 너의 모든 분석은 **오직 'INPUT DATA'에 제시된 숫자와 사실에만 기반**해야 한다.
    2.  **수치 인용 의무:** 주장을 증명하기 위해 **반드시 해당 수치를 괄호 안에 함께 제시**해야 한다. (예: '성장주({theme_perf.get('성장주', {}).get('change_percent', 0):.2f}%)는 가치주({theme_perf.get('가치주', {}).get('change_percent', 0):.2f}%) 대비 부진했습니다.')
    3.  **창작 금지:** **사실을 만들어내지 마라.** 데이터가 특정 내러티브와 다르다면, 그 '다름' 자체를 분석하라.
    4.  **시점 분석 (Timing Analysis):** 각 지표의 **'최종 업데이트'** 날짜를 반드시 확인하라. 만약 특정 지표의 날짜가 다른 지표들보다 과거에 멈춰있다면(예: 휴장일), 이를 '독자적 강세/약세'로 성급하게 해석해서는 안 된다. 대신, **'시점 불일치'**가 발생했음을 명확히 지적하고, 그로 인해 분석에 어떤 한계가 있는지 설명하라.
    5. **시계열 분석 의무:** **'핵심 경제 지표' 분석 시, 반드시 '12개월 추이' 숫자 데이터를 사용하여 추세(상승/하락/횡보)를 심층 분석**하고, 그 추세가 시장에 미치는 영향을 해석해야 한다. (예: '실업률(4.30) 자체는 낮지만, 12개월 추이가 [3.8, ..., 4.3]으로 꾸준히 상승하고 있어 고용 시장의 점진적인 둔화를 시사한다.'
    ---

    **INPUT DATA:**
    1. 최신 시장 지표 및 5일 추세:
    {data_summary_text}
    2. 최신 경제 뉴스 헤드라인:
    {news_summary_text}
    3. 최근 5일간 섹터별 자금 흐름(수익률):
    {sector_summary_text}
    4. 실시간 테마 & 자산군별 성과 및 '5일 추세':
    {theme_summary_text}
    5. 향후 2주 내 주요 이벤트:
    {event_summary_text}
    6. 핵심 경제 지표 (수치 | 상태 | 12개월 추이):
    {eco_summary_text}

    

    **MISSION:**
    위 '절대 규칙'을 반드시 준수하여, 모든 입력 데이터를 종합 분석하고 아래 JSON 형식에 맞춰 '요약'과 '상세 리포트'를 생성하라. 'report' 필드의 내용은 마크다운(Markdown) 형식으로 작성해야 한다.
    ** 7가지 모든 데이터를 종합 분석하라. **특히, 6번 '핵심 경제 지표'의 '12개월 추이'를 심층 분석하여 현재 경제가 확장 국면인지, 둔화 국면인지, 혹은 변곡점에 있는지 진단하라.** 시장의 종합적인 건강 상태를 평가하라.
    분석 시에는 각 테마의 '당일 성과'와 '5일 추세'를 함께 고려하여 시장의 성격을 심층 분석하라. (예: '반도체(-1.5%)는 오늘 하락했지만 5일 추세는 상승세이므로, 건강한 조정일 수 있다.')
    ** 현재 시장의 움직임이 5번 '주요 이벤트'를 앞둔 기대감 또는 경계감을 반영하고 있는지 해석하라.
    모든 입력 데이터를 분석하는 과정에서, 오늘 시장에 의미 있는 영향을 준 모든 '핵심 동인(Key Drivers)'들을 중요도 순으로 찾아내어 분석하라. '핵심 동인'이란, 특정 섹터/테마/자산군의 이례적인 움직임을 유발한 가장 유력한 뉴스나 이벤트를 의미한다.


    **OUTPUT JSON FORMAT:**
    {{
      "summary": "모든 분석을 압축한 가장 중요한 핵심 결론 한 문장",
      "report": "💡 **AI 시장 종합 진단**\\n\\n* **시장 현상 (What):** ...\\n* **내부 동력 (Why):** ...\\n* **핵심 인사이트:** ...\\n* **종합 코멘트:** ..."
    }}
    """

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        result_json = json.loads(response.text)
        return result_json
    except Exception as e:
        return {"summary": "AI 브리핑 생성 중 오류가 발생했습니다.", "report": f"**오류 발생:**\n\n```\n{str(e)}\n```"}
# --- [MOD v38.1 End] ---


# --- [MOD v35.6 Start] 섹터 성과 데이터 수집 함수 ---
@st.cache_data(ttl=1800) # 30분마다 데이터 갱신
def get_sector_performance():
    """11개 주요 섹터 ETF의 최근 5일간의 성과를 다운로드하여 반환합니다."""
    sector_tickers = {
        'Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF',
        'Consumer Discretionary': 'XLY', 'Communication Services': 'XLC', 'Industrials': 'XLI',
        'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Utilities': 'XLU',
        'Real Estate': 'XLRE', 'Materials': 'XLB'
    }
    try:
        data = yf.download(list(sector_tickers.values()), period="5d", progress=False)
        if data.empty:
            return pd.DataFrame()
        
        performance = {}
        for sector, ticker in sector_tickers.items():
            series = data['Close'][ticker].dropna()
            if len(series) >= 2:
                # 5일간의 누적 수익률 계산
                perf = (series.iloc[-1] / series.iloc[0] - 1) * 100
                performance[sector] = perf
        
        if not performance:
            return pd.DataFrame()

        perf_df = pd.DataFrame(list(performance.items()), columns=['Sector', 'Performance_5D'])
        return perf_df

    except Exception:
        return pd.DataFrame()
# --- [MOD v35.6 End] ---


# finance_core.py 파일의 get_economic_indicators 함수를 아래 코드로 전체 교체

@st.cache_data(ttl=43200) # 12시간마다 갱신
def get_economic_indicators():
    """[최종 고도화] 데이터 길이에 따라 '장기/단기 추세'를 자동으로 선택하여 분석합니다."""
    
    indicator_definitions = {
        "미국 장단기 금리차": {"code": "T10Y2Y", "type": "선행"},
        "개인소비지출(PCE)": {"code": "PCE", "type": "동행"},
        "실업률": {"code": "UNRATE", "type": "후행"},
        "근원 CPI (YoY)": {"code": "CORESTICKM159SFRBATL", "type": "후행"},
    }
    try:
        fred_api_key = st.secrets["FRED_API_KEY"]
        fred = Fred(api_key=fred_api_key)
        results = {}
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

        for name, info in indicator_definitions.items():
            series = fred.get_series(info['code'], observation_start=start_date)
            if not series.empty:
                series_monthly = series.resample('M').last().dropna()
                
                # [핵심] 데이터 길이에 따라 유연하게 추세 분석
                if len(series_monthly) >= 6:
                    # Case 1: 데이터가 충분하면 장기 추세 분석
                    avg_recent_3m = series_monthly.iloc[-3:].mean()
                    avg_prev_3m = series_monthly.iloc[-6:-3].mean()
                    trend_change = avg_recent_3m - avg_prev_3m
                    trend_label = "(3mo avg)"
                elif len(series_monthly) >= 2:
                    # Case 2: 데이터가 부족하면 단기 추세(MoM) 분석
                    latest_value_trend = series_monthly.iloc[-1]
                    prev_value_trend = series_monthly.iloc[-2]
                    trend_change = latest_value_trend - prev_value_trend
                    trend_label = "(MoM)"
                else:
                    # 추세 분석 불가
                    trend_change = 0
                    trend_label = ""

                latest_value = series_monthly.iloc[-1]
                
                if trend_change > 0.05: trend_icon = "🔼"
                elif trend_change < -0.05: trend_icon = "🔽"
                else: trend_icon = "⏺️"
                trend_text = f"{trend_icon} {trend_label}"

                # (이하 상태 평가 로직은 이전과 동일)
                status_text = "중립적 ⚪️"
                if name == "미국 장단기 금리차":
                    if latest_value > 0.25: status_text = "긍정적 🟢"
                    elif latest_value < 0: status_text = "부정적 🔴"
                # ... (다른 지표들의 상태 평가 로직)
                
                results[name] = {
                    "value": latest_value, "type": info['type'],
                    "trend": trend_text, "status": status_text,
                    "timeseries": series_monthly.tail(12).round(2).tolist()
                }
        return results
    except Exception as e:
        st.warning(f"FRED 경제 지표 로딩 중 오류: {e}")
        return {}

# [신규 기능] AI 포트폴리오 전략 브리핑 생성 함수
def generate_portfolio_briefing(summary_df, detail_df):
    """
    목표 대비 배분 현황과 개별 종목 상세 현황을 바탕으로
    AI가 포트폴리오의 종합 진단 및 전략 제안을 생성합니다.
    """
    model = get_gem_core_ai()

    # AI에게 전달할 데이터 요약
    summary_text = summary_df.to_string(index=False)
    
    # 건강검진 데이터 중 유의미한 정보만 필터링 (예: RSI > 70 또는 고점대비 < -30%)
    critical_stocks = detail_df[
        (detail_df['RSI'] > 70) | (detail_df['고점대비(%)'] < -30)
    ][['종목명', '자산티어', 'RSI', '고점대비(%)']].copy()
    critical_text = "현재 특별한 위험/기회 신호를 보이는 개별 종목은 없습니다."
    if not critical_stocks.empty:
        critical_text = "아래는 현재 유의미한 신호를 보이는 개별 종목 현황입니다:\n" + critical_stocks.to_string(index=False)

    prompt = f"""
    **SYSTEM ROLE:** 당신은 투자 전략가 'GEM: Finance'입니다. 당신의 임무는 MASTER의 포트폴리오 데이터를 분석하여, 현재 상태를 명확히 진단하고 다음 행동을 제안하는 것입니다. MASTER의 투자 철학은 '현명한 공격'이며, 비중이 부족한 자산을 저렴하게 매수하고, 과열된 자산은 일부 이익 실현하여 리밸런싱하는 것을 선호합니다.

    **INPUT DATA:**
    1. 목표 대비 자산 배분 현황:
    {summary_text}

    2. 주의가 필요한 개별 종목 현황 (RSI 과열 또는 과도한 하락):
    {critical_text}

    **MISSION:**
    위 두 가지 데이터를 종합하여, 아래 형식에 맞춰 'AI 포트폴리오 전략 브리핑'을 생성하십시오. 모든 내용은 한국어로 작성해야 합니다.

    ---
    ### 💡 AI 포트폴리오 전략 브리핑

    * **종합 진단:** (자산 배분 현황을 한 문장으로 요약하고, 개별 종목의 위험/기회 요소를 언급하며 현재 포트폴리오 상태를 진단합니다. 예: '현재 포트폴리오는 코어 비중이 부족한 상태이며, 동시에 핵심 위성 자산인 SMCI가 단기 과열 신호를 보이고 있습니다.')

    * **전략 제안:** (위 진단을 바탕으로 구체적인 리밸런싱 행동 계획을 제안합니다. 어떤 자산을 팔고, 어떤 자산을 사야 하는지 명확하게 제시하십시오. 예: '따라서, 단기 과열 상태인 SMCI의 일부를 이익 실현하고, 확보된 현금으로 비중이 부족한 코어 자산을 추가 매수하여 포트폴리오의 균형을 맞추는 리밸런싱을 고려할 최적의 시점입니다.')
    ---
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"**AI 브리핑 생성 중 오류 발생:**\n\n{e}"




# [신규] 다른 모든 데이터 함수에 대한 허브 경유 함수들

def create_hub_function(original_function, key_prefix, minutes_to_live=60):
    """
    반복적인 허브 함수 생성을 위한 팩토리 함수.
    """
    def hub_function(ticker):
        now = datetime.now()
        hub_key = f"{key_prefix}_{ticker}"
        
        if hub_key in st.session_state.data_hub:
            data, timestamp = st.session_state.data_hub[hub_key]
            if (now - timestamp) < timedelta(minutes=minutes_to_live):
                return data

        new_data = original_function(ticker)
        
        if new_data is not None: # 데이터가 None이 아닐 경우에만 저장
            st.session_state.data_hub[hub_key] = (new_data, now)
            
        return new_data
    return hub_function

# 각 원본 함수에 대해 허브 경유 함수 생성
get_profile_from_hub = create_hub_function(get_company_profile, "profile")
get_news_from_hub = create_hub_function(get_company_news, "news")
get_financials_from_hub = create_hub_function(get_basic_financials, "financials")
get_peers_from_hub = create_hub_function(get_company_peers, "peers")
get_earnings_from_hub = create_hub_function(get_company_earnings, "earnings")
get_calendar_from_hub = create_hub_function(get_earnings_calendar, "calendar")
get_candles_from_hub = create_hub_function(get_stock_candles, "candles")


@st.cache_data
def get_analyst_ratings(ticker):
    """Finnhub API를 사용하여 애널리스트 평가 데이터를 가져옵니다."""
    try:
        ratings = finnhub_client.recommendation_trends(ticker)
        if ratings:
            # 가장 최신 월의 데이터를 사용
            latest_rating = ratings[0]
            return {
                "period": latest_rating['period'],
                "strongBuy": latest_rating['strongBuy'],
                "buy": latest_rating['buy'],
                "hold": latest_rating['hold'],
                "sell": latest_rating['sell'],
                "strongSell": latest_rating['strongSell']
            }
    except Exception:
        return None
    return None

@st.cache_data
def get_insider_transactions(ticker):
    """Finnhub API를 사용하여 최근 90일간의 내부자 거래를 요약합니다."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        transactions = finnhub_client.stock_insider_transactions(ticker, _from=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'))
        
        if transactions and 'data' in transactions and transactions['data']:
            df = pd.DataFrame(transactions['data'])
            # 'mspr'는 총 거래 금액을 의미
            total_buy_value = df[df['change'] > 0]['mspr'].sum()
            total_sell_value = abs(df[df['change'] < 0]['mspr'].sum())
            net_value = total_buy_value - total_sell_value
            return {
                "totalBuyValue": total_buy_value,
                "totalSellValue": total_sell_value,
                "netValue": net_value
            }
    except Exception:
        return None
    return None

# 새로운 정보 수집 함수들을 중앙 허브에 등록
get_ratings_from_hub = create_hub_function(get_analyst_ratings, "ratings", minutes_to_live=1440) # 하루에 한번 갱신
get_insider_trans_from_hub = create_hub_function(get_insider_transactions, "insider", minutes_to_live=1440) # 하루에 한번 갱신


# [신규] '일상 건강검진'을 수행하는 통합 데이터 처리기
def get_health_check_data(ticker_list):
    """
    종목 리스트를 받아 '일상 건강검진'을 수행하고,
    결과를 데이터 허브에 캐시한 뒤 반환합니다.
    """
    summary_list = []
    
    # yfinance를 사용하여 모든 종목의 1년치 데이터를 한 번에 가져옴
    try:
        hist_data = yf.download(ticker_list, period="1y", progress=False)
        if hist_data.empty:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"주가 데이터 일괄 다운로드 실패: {e}")
        return pd.DataFrame()

    for ticker in ticker_list:
        hub_key = f"health_check_{ticker}"
        
        # 1. 캐시 확인 (1시간짜리 캐시)
        now = datetime.now()
        if hub_key in st.session_state.data_hub:
            data, timestamp = st.session_state.data_hub[hub_key]
            if (now - timestamp) < timedelta(hours=1):
                summary_list.append(data)
                continue # 캐시된 데이터 사용

        # 2. 데이터 계산 (캐시가 없을 경우)
        try:
            # MultiIndex에서 단일 종목 데이터 추출
            if len(ticker_list) > 1:
                hist = hist_data.loc[:, (slice(None), ticker)]
                hist.columns = hist.columns.droplevel(1)
            else:
                hist = hist_data
            
            if hist.empty or len(hist) < 20: continue

            # 핵심 지표 계산
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change_percent = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            high_52w = hist['High'].max()
            mdd_percent = ((current_price - high_52w) / high_52w) * 100 if high_52w != 0 else 0
            
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            volume_avg_20d = hist['Volume'].rolling(window=20).mean().iloc[-1]
            volume_change = (hist['Volume'].iloc[-1] / volume_avg_20d) * 100 if volume_avg_20d != 0 else 0

            ticker_summary = {
                "종목코드": ticker, "현재가": current_price, "등락률(%)": change_percent,
                "고점대비(%)": mdd_percent, "RSI": rsi, "거래량(%)": volume_change
            }

            # 3. 계산 결과를 캐시에 저장
            st.session_state.data_hub[hub_key] = (ticker_summary, now)
            summary_list.append(ticker_summary)

        except Exception:
            continue
            
    if not summary_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(summary_list)
    # 데이터 타입을 숫자로 명시적 변환
    numeric_cols = ["현재가", "등락률(%)", "고점대비(%)", "RSI", "거래량(%)"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

# [✨ NEW & REVISED] '종합 팩터 시트'를 생성하는 새로운 핵심 함수
def create_factor_sheet(ticker, latest_log, candles_df, news, ratings, insider_trans):
    """
    모든 변경점 팩터를 수집하고 AI에게 전달할 '종합 팩터 시트' 텍스트를 생성합니다.
    """
    if latest_log is None:
        return "과거 분석 기록이 없어 비교할 데이터가 없습니다. 첫 분석을 수행합니다."

    sheet = [f"**[종합 팩터 시트: {ticker}]**"]
    
    # 1. 과거 데이터 로드
    try:
        past_data = json.loads(latest_log.get('주요 데이터', '{}'))
        past_quote = past_data.get('quote', {})
        past_price = past_quote.get('c', 0)
    except Exception:
        past_price = 0

    # 2. 시장 데이터 팩터
    if not candles_df.empty:
        current_price = candles_df['Close'].iloc[-1]
        market_factors = []
        if past_price > 0:
            price_change_cum = ((current_price - past_price) / past_price) * 100
            market_factors.append(f"- 누적 변동률: {price_change_cum:+.2f}% (vs. 마지막 분석)")
        
        price_change_recent = ((current_price - candles_df['Close'].iloc[-2]) / candles_df['Close'].iloc[-2]) * 100
        market_factors.append(f"- 최근 변동률: {price_change_recent:+.2f}% (vs. 전일)")

        if len(candles_df) > 20:
            avg_volume_20d = candles_df['Volume'].rolling(window=20).mean().iloc[-2] # 전일까지의 평균
            current_volume = candles_df['Volume'].iloc[-1]
            if avg_volume_20d > 0:
                volume_change_percent = (current_volume / avg_volume_20d) * 100
                market_factors.append(f"- 거래량: 20일 평균 대비 {volume_change_percent:.0f}% 수준")
        
        if market_factors:
            sheet.append("\n**I. 시장 데이터**\n" + "\n".join(market_factors))

    # 3. 기술적 상태 팩터 (candles_df는 이미 기술적 지표가 추가된 상태로 전달받아야 함)
    if not candles_df.empty and 'SMA20' in candles_df.columns:
        tech_factors = []
        latest = candles_df.iloc[-1]
        previous = candles_df.iloc[-2]
        # RSI 상태 변화
        if previous['RSI14'] < 70 and latest['RSI14'] >= 70: tech_factors.append("- RSI 과매수 구간(70) 진입")
        elif previous['RSI14'] > 30 and latest['RSI14'] <= 30: tech_factors.append("- RSI 과매도 구간(30) 진입")
        # MACD 신호
        if previous['MACD'] < previous['SignalLine'] and latest['MACD'] > latest['SignalLine']: tech_factors.append("- MACD 골든 크로스 발생")
        elif previous['MACD'] > previous['SignalLine'] and latest['MACD'] < latest['SignalLine']: tech_factors.append("- MACD 데드 크로스 발생")
        
        if tech_factors:
            sheet.append("\n**II. 기술적 상태**\n" + "\n".join(tech_factors))

    # 4. 기업 및 외부 환경 팩터
    env_factors = []
    if news:
        past_news_headlines = past_data.get('news_headlines', [])
        current_news_headlines = [item['headline'] for item in news[:5]]
        new_headlines = [h for h in current_news_headlines if h not in past_news_headlines]
        if new_headlines:
            env_factors.append(f"- {len(new_headlines)}개의 신규 주요 뉴스 발생")

    if ratings:
        buy_ratings = ratings.get('strongBuy', 0) + ratings.get('buy', 0)
        total_ratings = buy_ratings + ratings.get('hold', 0) + ratings.get('sell', 0) + ratings.get('strongSell', 0)
        if total_ratings > 0:
            buy_ratio = (buy_ratings / total_ratings) * 100
            env_factors.append(f"- 최신 애널리스트 '매수' 의견: {buy_ratio:.0f}% ({total_ratings}명 참여)")

    if insider_trans and insider_trans.get('netValue') != 0:
        if insider_trans['netValue'] > 0:
            env_factors.append(f"- 최근 90일간 내부자 순매수: 약 ${insider_trans['netValue']:,.0f}")
        else:
            env_factors.append(f"- 최근 90일간 내부자 순매도: 약 ${abs(insider_trans['netValue']):,.0f}")

    if env_factors:
        sheet.append("\n**III. 기업 및 외부 환경**\n" + "\n".join(env_factors))

    if len(sheet) == 1: # 아무 팩터도 추가되지 않았다면
        return "마지막 분석 이후 유의미한 데이터 변경점이 감지되지 않았습니다."
        
    return "\n".join(sheet)


# --- 4. 메인 UI 및 로직 ---
st.title("💎 GEM: Finance Dashboard")
st.caption("v34.0 - Final Strategy Implemented")

# [수정] 모든 세션 상태를 여기서 한 번에 초기화
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.active_view = "💼 포트폴리오"
    st.session_state.data_hub = {}
    # ... 기타 초기화 필요한 session_state ...

if not st.session_state.data_loaded:
    with st.spinner("Initializing System... Loading data from Google Sheets..."):
        st.session_state.portfolio_df, st.session_state.watchlist_df, st.session_state.cash_df = load_data_from_gsheet()
    st.session_state.data_loaded = True
    st.rerun() # 데이터를 로드한 후, 기본 뷰를 제대로 표시하기 위해 한 번 더 재실행

# --- 포트폴리오 뷰 ---
# --- 포트폴리오 뷰 ---
elif st.session_state.active_view == "💼 포트폴리오":
    st.header("💼 Portfolio Command Center")
    
    portfolio_df = st.session_state.portfolio_df
    cash_df = st.session_state.cash_df

    if not cash_df.empty:
        cash_df = cash_df[~cash_df['종목명'].str.contains('비상금', na=False)]

    if not portfolio_df.empty or not cash_df.empty:
        # --- 1. 모든 데이터 준비 및 최종 데이터프레임 생성 ---
        all_tickers_for_price = portfolio_df['종목코드'].dropna().unique().tolist()
        if all_tickers_for_price:
            current_prices, usd_krw_rate = get_current_prices_and_rate(all_tickers_for_price)
            st.sidebar.metric("USD/KRW 환율", f"₩{usd_krw_rate:,.2f}")
            invest_dashboard_df = create_portfolio_dashboard(portfolio_df, current_prices, usd_krw_rate)
        else:
            invest_dashboard_df = pd.DataFrame()

        health_check_df = pd.DataFrame()
        if not invest_dashboard_df.empty:
            all_investment_tickers = invest_dashboard_df['종목코드'].dropna().unique().tolist()
            if all_investment_tickers:
                with st.spinner("개별 자산 건강검진 수행 중..."):
                    health_check_df = get_health_check_data(all_investment_tickers)
            if not health_check_df.empty:
                invest_dashboard_df = pd.merge(invest_dashboard_df, health_check_df, on='종목코드', how='left')

        cash_dashboard_df = pd.DataFrame()
        if not cash_df.empty:
            cash_dashboard_df = cash_df.rename(columns={'금액(KRW)': '현재 평가 금액 (KRW)'})
            cash_dashboard_df['수익률 (%)'] = 0; cash_dashboard_df['손익 (고유)'] = 0; cash_dashboard_df['수량'] = '-';
            cash_dashboard_df['평균 단가 (고유)'] = '-'; cash_dashboard_df['현재가 (고유)'] = '-';

        final_dashboard_df = pd.concat([invest_dashboard_df, cash_dashboard_df], ignore_index=True)
        
        final_dashboard_df['계좌구분'] = (
            final_dashboard_df['계좌구분']
            .astype(str).str.strip()
            .replace({'nan': '현금', 'None': '현금', '': '현금'}).str.lower()
        )
        final_dashboard_df['계좌구분'] = final_dashboard_df['계좌구분'].replace({
            'irp': 'IRP', '회사퇴직dc': '회사퇴직DC', 'isa': 'ISA',
            '해외 직투': '해외 직투', '연금저축': '연금저축', '현금': '현금'
        })

        # --- 2. AI 브리핑을 위한 데이터 준비 ---
        direct_investment_df = final_dashboard_df[final_dashboard_df['계좌구분'] == '해외 직투'].copy()
        total_direct_investment_value = direct_investment_df['현재 평가 금액 (KRW)'].sum()
        targets = {"코어 (Tier 1+1.5)": 60.0, "핵심 위성 (Tier 2)": 30.0, "테마형 위성 (Tier 3)": 10.0}
        summary_df_for_ai = pd.DataFrame()

        if total_direct_investment_value > 0 and '자산티어' in direct_investment_df.columns:
            core_value = direct_investment_df[direct_investment_df['자산티어'].isin(['Tier 1', 'Tier 1.5'])]['현재 평가 금액 (KRW)'].sum()
            tier2_value = direct_investment_df[direct_investment_df['자산티어'] == 'Tier 2']['현재 평가 금액 (KRW)'].sum()
            tier3_value = direct_investment_df[direct_investment_df['자산티어'] == 'Tier 3']['현재 평가 금액 (KRW)'].sum()
            current_alloc = {
                "코어 (Tier 1+1.5)": (core_value / total_direct_investment_value) * 100,
                "핵심 위성 (Tier 2)": (tier2_value / total_direct_investment_value) * 100,
                "테마형 위성 (Tier 3)": (tier3_value / total_direct_investment_value) * 100,
            }
            summary_data_for_ai = []
            for group, target_pct in targets.items():
                current_pct = current_alloc.get(group, 0)
                diff = current_pct - target_pct
                status = "정상"
                if diff < -5: status = "부족"
                elif diff > 5: status = "과다"
                summary_data_for_ai.append({
                    "자산 그룹": group, "현재 비중(%)": current_pct, "목표 비중(%)": target_pct,
                    "차이(%)": diff, "상태 평가": status
                })
            summary_df_for_ai = pd.DataFrame(summary_data_for_ai)
        
        # --- 3. [신규] AI 포트폴리오 전략 브리핑 표시 ---
        if not summary_df_for_ai.empty:
            with st.spinner("AI가 포트폴리오를 분석하고 전략을 제안하는 중..."):
                briefing = generate_portfolio_briefing(summary_df_for_ai, final_dashboard_df)
                st.markdown(briefing)

        # --- 2. 계좌별 성과 요약 ---
        st.subheader("📊 성과 요약")

        total_value = final_dashboard_df['현재 평가 금액 (KRW)'].sum()
        total_invest_df = final_dashboard_df[final_dashboard_df['계좌구분'] != '현금']
        total_cost = total_invest_df['총 매수 금액 (KRW)'].sum() if not total_invest_df.empty else 0
        total_pl = total_invest_df['손익 (KRW)'].sum() if not total_invest_df.empty else 0
        total_pl_percent = (total_pl / total_cost) * 100 if total_cost > 0 else 0
        
        with st.container(border=True):
            st.markdown("##### 총 자산 (전체 계좌, 비상금 제외)")
            cols = st.columns(3)
            cols[0].metric("총 평가 자산", f"₩{total_value:,.0f}")
            cols[1].metric("총 손익 (투자)", f"₩{total_pl:,.0f}", f"{total_pl_percent:.2f}%")
            cols[2].metric("총 투자 원금", f"₩{total_cost:,.0f}")

        with st.expander("계좌별 상세 성과 보기", expanded=True):
            # (이하 로직은 변경 없음, 이제 정상적으로 작동)
            account_summary_data = []
            accounts = final_dashboard_df['계좌구분'].unique()
            for account in accounts:
                account_df = final_dashboard_df[final_dashboard_df['계좌구분'] == account]
                account_invest_df = account_df[account_df['계좌구분'] != '현금']
                acc_total_value = account_df['현재 평가 금액 (KRW)'].sum()
                acc_total_cost = account_invest_df['총 매수 금액 (KRW)'].sum() if not account_invest_df.empty else 0
                acc_total_pl = account_invest_df['손익 (KRW)'].sum() if not account_invest_df.empty else 0
                acc_total_pl_percent = (acc_total_pl / acc_total_cost) * 100 if acc_total_cost > 0 else 0
                
                account_summary_data.append({
                    "계좌 구분": account, "평가 자산": acc_total_value,
                    "손익": acc_total_pl, "수익률(%)": acc_total_pl_percent,
                    "투자 원금": acc_total_cost
                })
            
            if account_summary_data:
                account_summary_df = pd.DataFrame(account_summary_data)
                account_order = ['해외 직투', 'ISA', '연금저축', 'IRP', '회사퇴직DC', '현금']
                ordered_categories = [acc for acc in account_order if acc in account_summary_df['계좌 구분'].values]
                account_summary_df['계좌 구분'] = pd.Categorical(account_summary_df['계좌 구분'], categories=ordered_categories, ordered=True)
                account_summary_df = account_summary_df.sort_values('계좌 구분').reset_index(drop=True)

                def color_return(val):
                    color = '#4CAF50' if val > 0 else '#F44336' if val < 0 else '#333333'
                    return f'color: {color}'
                
                st.dataframe(account_summary_df.style.format({
                                 "평가 자산": "₩{:,.0f}", "손익": "₩{:,.0f}",
                                 "수익률(%)": "{:,.2f}%", "투자 원금": "₩{:,.0f}"
                             }).applymap(color_return, subset=['수익률(%)']), 
                             use_container_width=True)
        st.divider()

        # --- 3. 목표 대비 자산 배분 현황 (해외 직투) ---
        st.subheader("🎯 목표 대비 자산 배분 현황 (해외 직투)")
        direct_investment_df = final_dashboard_df[final_dashboard_df['계좌구분'] == '해외 직투'].copy()
        total_direct_investment_value = direct_investment_df['현재 평가 금액 (KRW)'].sum()
        targets = {"코어 (Tier 1+1.5)": 60.0, "핵심 위성 (Tier 2)": 30.0, "테마형 위성 (Tier 3)": 10.0}

        if total_direct_investment_value > 0 and '자산티어' in direct_investment_df.columns:
            core_value = direct_investment_df[direct_investment_df['자산티어'].isin(['Tier 1', 'Tier 1.5'])]['현재 평가 금액 (KRW)'].sum()
            tier2_value = direct_investment_df[direct_investment_df['자산티어'] == 'Tier 2']['현재 평가 금액 (KRW)'].sum()
            tier3_value = direct_investment_df[direct_investment_df['자산티어'] == 'Tier 3']['현재 평가 금액 (KRW)'].sum()
            current_alloc = {
                "코어 (Tier 1+1.5)": (core_value / total_direct_investment_value) * 100,
                "핵심 위성 (Tier 2)": (tier2_value / total_direct_investment_value) * 100,
                "테마형 위성 (Tier 3)": (tier3_value / total_direct_investment_value) * 100,
            }
            summary_data = []
            for group, target_pct in targets.items():
                current_pct = current_alloc.get(group, 0)
                diff = current_pct - target_pct
                status = "🟢 정상"
                if diff < -5: status = "🔴 부족"
                elif diff > 5: status = "🟡 과다"
                summary_data.append({
                    "자산 그룹": group, "현재 비중(%)": current_pct, "목표 비중(%)": target_pct,
                    "차이(%)": diff, "상태 평가": status
                })
            summary_df = pd.DataFrame(summary_data)
            def color_status(val):
                color = 'red' if '부족' in val else 'orange' if '과다' in val else 'green'
                return f'color: {color}'
            st.dataframe(summary_df.style
                         .format({"현재 비중(%)": "{:.1f}%", "목표 비중(%)": "{:.1f}%", "차이(%)": "{:+.1f}%"})
                         .applymap(color_status, subset=['상태 평가'])
                         .bar(subset=["현재 비중(%)"], color='lightblue', vmin=0, vmax=100),
                         use_container_width=True)
        else:
            st.warning("'해외 직투' 계좌에 자산이 없거나 '자산티어'가 지정되지 않아 목표 대비 현황을 분석할 수 없습니다.")
        
        st.divider()
        
         # --- 4. 보유 자산 상세 및 자산 배분 차트 ---
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.subheader("보유 자산 상세 (전체)")
            display_cols = ['계좌구분', '종목명', '자산티어', '수량', '평균 단가 (고유)', '현재가 (고유)',  
                            '손익 (고유)', '수익률 (%)', '현재 평가 금액 (KRW)', 
                            '고점대비(%)', 'RSI', '거래량(%)']
            
            formatter = {'손익 (고유)': '{:,.2f}', '수익률 (%)': '{:.2f}%', '현재 평가 금액 (KRW)': '₩{:,.0f}', 
                         "고점대비(%)": "{:,.2f}%", "RSI": "{:.1f}", "거래량(%)": "{:,.0f}%"}
            
            # ✨ [핵심 수정] 불필요한 데이터 처리 로직 삭제, 최종 데이터프레임을 바로 사용
            st.dataframe(final_dashboard_df.style
                         .format(formatter, na_rep="-")
                         .background_gradient(cmap='RdYlGn', subset=['수익률 (%)'])
                         .bar(subset=['고점대비(%)'], color='#FFA07A', vmin=-100, vmax=0)
                         .bar(subset=['RSI'], align='mid', color=['#d65f5f', '#5fba7d'], vmin=0, vmax=100)
                         .bar(subset=['거래량(%)'], color='lightblue'), 
                         use_container_width=True,
                         column_order=display_cols # 컬럼 순서 지정
            )
        
        with col2:
            st.subheader("자산 배분 (전체)")
            chart_group_by = st.radio("차트 기준", ['자산티어', '계좌구분'], horizontal=True, key='chart_group')
            filter_cols = st.columns(2)
            exclude_base = filter_cols[0].checkbox("'기반' 티어 제외", value=True)
            exclude_cash = filter_cols[1].checkbox("'현금' 자산 제외", value=True)
            
            chart_df = final_dashboard_df.copy()
            
            if exclude_base and '자산티어' in chart_df.columns:
                chart_df = chart_df[~chart_df['자산티어'].str.contains('기반', na=False)]
            if exclude_cash and '자산티어' in chart_df.columns:
                chart_df = chart_df[~chart_df['자산티어'].str.contains('현금', na=False)]
            
            if not chart_df.empty and chart_df['현재 평가 금액 (KRW)'].sum() > 0:
                allocation = chart_df.groupby(chart_group_by)['현재 평가 금액 (KRW)'].sum()
                fig_tier = px.pie(values=allocation.values, names=allocation.index, title=f"{chart_group_by}별 비중", hole=.3)
                st.plotly_chart(fig_tier, use_container_width=True)
            else:
                st.warning("차트에 표시할 데이터가 없습니다.")
    else:
        st.info("Google Sheets에서 포트폴리오 또는 현금 데이터를 불러올 수 없습니다.")


# --- [추가] 레이더 뷰 ---
elif st.session_state.active_view == "📡 레이더":
    st.header("📡 Stock Radar")

    watchlist_df = st.session_state.watchlist_df

    if watchlist_df.empty or '종목코드' not in watchlist_df.columns:
        st.info("관심 종목이 없습니다. Google Sheets의 'Watchlist'에 종목을 추가해주세요.")
    else:
        tickers = watchlist_df['종목코드'].dropna().unique().tolist()

        # [수정] 스피너 메시지를 현재 작업에 맞게 변경
        with st.spinner("레이더 데이터를 스캔하는 중... (일상 건강검진 수행)"):
            # [수정] 새로운 '통합 건강검진 모듈'을 호출하도록 변경
            radar_df = get_health_check_data(tickers)

        if not radar_df.empty:
            # [변경 없음] 이하 데이터프레임 스타일링 및 표시 코드는 기존과 동일합니다.
            formatter = {
                "등락률(%)": "{:,.2f}%",
                "고점대비(%)": "{:,.2f}%",
                "RSI": "{:.1f}",
                "거래량(%)": "{:,.0f}%",
                "현재가": lambda x: f"${x:,.2f}" # 간단하게 달러로 통일 (추후 고도화 가능)
            }

            st.dataframe(radar_df.style
                .format(formatter)
                .background_gradient(cmap='RdYlGn', subset=['등락률(%)'])
                .bar(subset=['고점대비(%)'], color='#FFA07A')
                .bar(subset=['RSI'], align='mid', color=['#d65f5f', '#5fba7d'])
                .bar(subset=['거래량(%)'], color='lightblue'),
                use_container_width=True
            )
        else:
            st.error("레이더 데이터를 가져오는 데 실패했습니다.")

elif st.session_state.active_view == "🔭 시장 건강 상태":
    st.header("🔭 Market Health Dashboard")

    # --- 1. 모든 데이터 준비 ---
    with st.spinner("시장 데이터 및 이벤트 정보를 수집 중..."):
        market_data, hist_data = get_market_status_data()
        sector_perf_df = get_sector_performance()
        theme_perf = get_theme_etf_performance()

        # --- 신규 데이터 로드 ---
        eco_indicators = get_economic_indicators()
        
        # 이벤트 데이터 준비
        eco_events = get_economic_calendar()
        portfolio_tickers = st.session_state.portfolio_df['종목코드'].dropna().unique().tolist()
        watchlist_tickers = st.session_state.watchlist_df['종목코드'].dropna().unique().tolist()
        all_my_tickers = list(set(portfolio_tickers + watchlist_tickers))
        earnings_events = get_portfolio_earnings_calendar(all_my_tickers)

        combined_events = []
        for event in eco_events:
            combined_events.append({'date': event['time'].split(' ')[0], 'type': 'eco', 'data': event})
        for event in earnings_events:
            combined_events.append({'date': event['date'], 'type': 'earn', 'data': event})
        sorted_events = sorted(combined_events, key=lambda x: x['date'])

    # --- 2. AI 종합 분석 (모든 데이터 준비 완료 후 실행) ---
    with st.spinner("AI가 시장 건강 상태 및 미래 이벤트를 종합 분석 중입니다..."):
        hub_key = "market_briefing_v4"
        now = datetime.now()
        if hub_key in st.session_state.data_hub and (now - st.session_state.data_hub[hub_key][1]) < timedelta(minutes=5):
            briefing_result = st.session_state.data_hub[hub_key][0]
        else:
            briefing_result = generate_market_health_briefing(market_data, hist_data, sector_perf_df, theme_perf, sorted_events, eco_indicators)
            st.session_state.data_hub[hub_key] = (briefing_result, now)

    # --- 3. UI 표시 ---
    # AI 브리핑 표시
    summary = briefing_result.get("summary", "AI가 요약을 생성하지 못했습니다.")
    full_report = briefing_result.get("report", "상세 리포트를 생성하지 못했습니다.")
    st.subheader("💡 AI 종합 진단 요약")
    st.info(summary)
    with st.expander("상세 분석 리포트 보기"):
        st.markdown(full_report)
    # --- [MOD v36.6 End] ---

    if market_data:
        st.divider()

         # --- [MOD v46 Start] 거시 지표 섹션 UI 재구성 ---
        st.subheader("🔭 시장 종합 계기판")

        # 2. 핵심 경제 지표 (선행/동행/후행)
        st.markdown("##### 🔑 핵심 경제 지표")

        eco_indicators = get_economic_indicators()
        if eco_indicators:
            cols = st.columns(len(eco_indicators))
            for i, (name, data) in enumerate(eco_indicators.items()):
                
                # [고도화] 상태(status)에 따라 델타 색상 결정
                delta_color = "off"
                if "긍정적" in data['status']: delta_color = "normal"
                elif "부정적" in data['status']: delta_color = "inverse"
                    
                cols[i].metric(
                    label=f"{name} ({data['type']})",
                    value=f"{data['value']:,.2f}",
                    delta=f"{data['status']} {data['trend']}", # 상태와 추세를 함께 표시
                    delta_color=delta_color
                )
        else:
            st.warning("핵심 경제 지표를 불러오는 데 실패했습니다.")

        
        st.divider()
        
        # 1. 거시 경제 지표 섹션 (기존과 동일)
        st.subheader("🌎 거시 경제 지표")
        # ... (거시 지표 표시하는 cols, indices, for 루프 등은 이전과 동일하게 유지) ...
        cols = st.columns(5)
        indices = ["S&P 500", "Nasdaq", "VIX", "US 10Y", "USD/KRW"]
        for i, name in enumerate(indices):
            if name in market_data and market_data[name]["price"] != "N/A":
                d = market_data[name]
                cols[i].metric(label=name, value=f"{d['price']:,.2f}", delta=f"{d['change']:,.2f} ({d['change_percent']:.2f}%)" if name not in ['VIX', 'US 10Y'] else f"{d['change']:.2f}")
        cols = st.columns(4)
        indices = ["Dollar", "Crude Oil", "Gold", "KOSPI"]
        for i, name in enumerate(indices):
             if name in market_data and market_data[name]["price"] != "N/A":
                d = market_data[name]
                cols[i].metric(label=name, value=f"{d['price']:,.2f}", delta=f"{d['change']:.2f} ({d['change_percent']:.2f}%)" if name not in ['Dollar'] else f"{d['change']:.2f}")

        st.divider()

        # 2. 자금 흐름 분석 섹션 (탭으로 분리)
        st.subheader("📊 자금 흐름 분석")
        tab1, tab2 = st.tabs(["전체 섹터 흐름 (11 Sectors)", "핵심 테마 & 자산군"])

        with tab1:
            # ✨ [복원] 11개 섹터 히트맵
            st.markdown("###### 5일 누적 수익률 기준")
            sector_perf_df = get_sector_performance()
            if not sector_perf_df.empty:
                fig = px.treemap(sector_perf_df, 
                                 path=[px.Constant("S&P 500 Sectors"), 'Sector'], 
                                 values=sector_perf_df['Performance_5D'].abs(),
                                 color='Performance_5D',
                                 color_continuous_scale=['#d65f5f', 'lightgray', '#5fba7d'],
                                 color_continuous_midpoint=0,
                                 custom_data=['Performance_5D'])
                fig.update_traces(texttemplate='%{label}<br>%{customdata[0]:.2f}%')
                fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("섹터 성과 데이터를 가져오는 데 실패했습니다.")

        with tab2:
            # ✨ [재구성] 테마 & 자산군 대시보드
            st.markdown("###### 당일 등락률 기준 (vs S&P 500)")
            theme_perf = get_theme_etf_performance()
            if theme_perf:
                # ... (이하 테마 ETF를 표시하는 display_etf_metric 함수 및 UI 로직은 이전과 동일) ...
                spy_perf_change = theme_perf.get("S&P 500", {}).get('change_percent', 0)
                def display_etf_metric(col, theme_name):
                    perf_data = theme_perf.get(theme_name, {})
                    perf_change = perf_data.get('change_percent', 0)
                    trend = perf_data.get('trend', '횡보')
                    trend_icon = "🔼" if trend == "상승" else "🔽" if trend == "하락" else "⏺️"
                    delta_vs_spy = perf_change - spy_perf_change
                    col.metric(
                        label=f"{trend_icon} {theme_name}",
                        value=f"{perf_change:.2f}%",
                        delta=f"{delta_vs_spy:.2f}% vs SPY",
                        delta_color="off" if abs(delta_vs_spy) < 0.01 else ("normal")
                    )
                st.markdown("<h6>1. 시장 성격 (Style)</h6>", unsafe_allow_html=True)
                cols = st.columns(2); display_etf_metric(cols[0], "가치주"); display_etf_metric(cols[1], "성장주")
                st.markdown("<h6>2. 핵심 기술 (Core Tech)</h6>", unsafe_allow_html=True)
                cols = st.columns(2); display_etf_metric(cols[0], "반도체"); display_etf_metric(cols[1], "AI")
                st.markdown("<h6>3. 미래 테마 (Future Forward)</h6>", unsafe_allow_html=True)
                cols = st.columns(3); display_etf_metric(cols[0], "로보틱스"); display_etf_metric(cols[1], "바이오테크"); display_etf_metric(cols[2], "차세대 전력")
                st.markdown("<h6>4. 방어 & 인컴 (Defense & Income)</h6>", unsafe_allow_html=True)
                cols = st.columns(2); display_etf_metric(cols[0], "고배당"); display_etf_metric(cols[1], "장기채")
                st.markdown("<h6>5. 시장 심리 (Sentiment)</h6>", unsafe_allow_html=True)
                cols = st.columns(2); display_etf_metric(cols[0], "혁신기술"); display_etf_metric(cols[1], "비트코인")
            else:
                st.warning("테마/자산군 ETF 성과 데이터를 가져오는 데 실패했습니다.")

        # --- [MOD v40 End] ---
        
        st.divider()
        st.subheader("🗓️ 향후 2주 주요 이벤트")

        # 데이터 로드
        eco_events = get_economic_calendar()
        
        # 포트폴리오 및 관심종목 티커 목록 가져오기
        portfolio_tickers = st.session_state.portfolio_df['종목코드'].dropna().unique().tolist()
        watchlist_tickers = st.session_state.watchlist_df['종목코드'].dropna().unique().tolist()
        all_my_tickers = list(set(portfolio_tickers + watchlist_tickers))
        
        earnings_events = get_portfolio_earnings_calendar(all_my_tickers)

        # 두 이벤트 리스트를 날짜 기준으로 통합하고 정렬
        combined_events = []
        for event in eco_events:
            combined_events.append({'date': event['time'].split(' ')[0], 'type': 'eco', 'data': event})
        for event in earnings_events:
            combined_events.append({'date': event['date'], 'type': 'earn', 'data': event})
            
        sorted_events = sorted(combined_events, key=lambda x: x['date'])

        if sorted_events:
            cols = st.columns(2)
            col_idx = 0
            
            for event in sorted_events:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d').strftime('%m/%d (%a)')
                
                # 좌우 컬럼에 번갈아 가며 이벤트 표시
                with cols[col_idx % 2]:
                    if event['type'] == 'eco':
                        impact = event['data']['impact']
                        impact_icon = "🔴" if impact == 'high' else "🟠" if impact == 'medium' else ""
                        st.markdown(f"**{event_date}**: {impact_icon} {event['data']['event']}")
                    elif event['type'] == 'earn':
                        st.markdown(f"**{event_date}**: 📢 Earnings - **${event['data']['ticker']}**")
                col_idx += 1
        else:
            st.info("향후 2주 내에 예정된 주요 이벤트가 없습니다.")
        # --- [MOD v41 End] ---
        st.subheader("주요 경제 뉴스")
        if market_data.get('news'):
            for item in market_data['news']:
                news_date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
                st.markdown(f"**[{item['headline']}]({item['url']})** - *{news_date}, {item['source']}*")
        else:
            st.warning("주요 경제 뉴스를 불러올 수 없습니다.")
    else:
        st.error("시장 현황 데이터를 가져오는 데 실패했습니다.")


elif st.session_state.active_view == "📡 탐색":
    st.header("📡 Discovery Engine: Status Tracker")
    st.info("선택한 시장의 전체 종목에 대한 기술적 상태를 분석하고, 한눈에 파악할 수 있도록 상태 태그를 부여합니다.")

    # --- 상태(Status) 판정 기준 설명 ---
    with st.expander("ℹ️ 상태(Status) 판정 기준"):
        st.markdown("""
        - **매수 신호 🟢**: 골든크로스가 발생했거나, RSI가 30 이하 과매도 구간에 진입한 종목.
        - **주의/매도 신호 🔴**: 데드크로스가 발생했거나, RSI가 70 이상 과매수 구간에 진입한 종목.
        - **중립/관망 ⚪️**: 위 신호에 해당하지 않는 나머지 모든 종목.
        """)

    ticker_lists = get_all_ticker_lists()
    
    if not ticker_lists:
        st.warning("'Tickers' 시트를 찾을 수 없거나 비어있습니다. Google Sheets를 확인해주세요.")
    else:
        selected_list_name = st.selectbox("탐색 대상 시장 선택:", list(ticker_lists.keys()))
        
        if st.button(f"🚀 {selected_list_name} 전체 종목 상태 분석", use_container_width=True, type="primary"):
            tickers_to_scan = ticker_lists.get(selected_list_name, [])
            st.session_state.screener_results = run_status_screener(tickers_to_scan)
            st.rerun()

    if "screener_results" in st.session_state:
        st.divider()
        results_df = st.session_state.screener_results
        
        st.subheader(f"📊 상태 분석 결과: {len(results_df)}개 종목")

        if not results_df.empty:
            # 등락률(%) 값에 따라 폰트 색상을 변경하는 함수
            def style_change_percent(val):
                color = 'red' if val < 0 else 'green' if val > 0 else '#525252' # 회색
                return f'color: {color}'

            st.dataframe(
                results_df.style
                    .applymap(style_change_percent, subset=['등락률(%)'])
                    .format({
                        "현재가": "${:,.2f}",
                        "등락률(%)": "{:+.2f}%",
                        "고점대비(%)": "{:.2f}%",
                        "RSI": "{:.1f}",
                    }),
                hide_index=True,
                use_container_width=True,
                height=800
            )
        else:
            st.error("데이터를 분석하는 데 실패했습니다. 티커 목록을 확인해주세요.")



# --- 상세 분석 뷰 ---
elif st.session_state.active_view == "🔍 상세 분석":
    if 'analysis_tickers' in st.session_state and st.session_state.analysis_tickers:
        main_ticker = st.session_state.analysis_tickers[0]
        st.header(f"🔍 {main_ticker} 상세 분석")

        # --- 1. 기초 데이터 로드 ---
        with st.spinner(f"'{main_ticker}' 상세 데이터를 가져오는 중..."):
            profile = get_profile_from_hub(main_ticker)
            quote = get_quote_from_hub(main_ticker)
            news = get_news_from_hub(main_ticker)
            financials_df = get_financials_from_hub(main_ticker)
            peers = get_peers_from_hub(main_ticker)
            earnings_data = get_earnings_from_hub(main_ticker)
            next_earnings_date = get_calendar_from_hub(main_ticker)
            candles_df = get_candles_from_hub(main_ticker)

        # --- 2. 탭 UI 생성 ---
        analysis_tab_names = ["💎 종합 진단", "📈 기술적 분석", "💰 펀더멘털", "📰 뉴스 및 개요", "📜 과거 분석 기록"]
        diag_tab, tech_tab, fin_tab, news_tab, log_tab = st.tabs(analysis_tab_names)

        # [핵심 수정] '단일 진실 공급원' 로직 강화
        # Ticker 변경 시, 이전 분석 결과 초기화
        if 'current_ticker' not in st.session_state or st.session_state.current_ticker != main_ticker:
            st.session_state.current_ticker = main_ticker
            st.session_state.last_analysis_text = None
            st.session_state.structured_reco = {}
            # ✨ NEW: Ticker 변경 시 최신 로그를 불러와 세션 상태 초기화
            with st.spinner(f"'{main_ticker}'의 최근 분석 기록을 불러오는 중..."):
                analysis_logs = load_analysis_log(main_ticker)
                if not analysis_logs.empty:
                    latest_log = analysis_logs.iloc[0]
                    st.session_state.last_analysis_text = latest_log.get('전체 분석 내용')
                    st.session_state.last_analysis_ticker = main_ticker
                st.rerun() # 최신 로그를 반영하여 화면을 다시 그립니다.

                        # --- 3. 각 탭 내용 구성 ---
# --- 3. 각 탭 내용 구성 ---
        with diag_tab:
            st.subheader(f"💎 {main_ticker} 종합 진단")
            cols = st.columns(4)
            if quote and quote.get('c') != 0:
                cols[0].metric("현재가", f"${quote.get('c', 0):.2f}", f"{quote.get('d', 0):.2f}$ ({quote.get('dp', 0):.2f}%)")
            if not candles_df.empty:
                candles_df_tech_diag = add_technical_indicators(candles_df.copy())
                if 'RSI14' in candles_df_tech_diag.columns and not pd.isna(candles_df_tech_diag['RSI14'].iloc[-1]):
                    cols[1].metric("RSI (14일)", f"{candles_df_tech_diag['RSI14'].iloc[-1]:.2f}")
                high_52w = candles_df['High'].max()
                if high_52w > 0:
                    cols[2].metric("52주 고점 대비", f"{((quote.get('c', 0) - high_52w) / high_52w) * 100:.2f}%")
            if profile:
                cols[3].metric("시가총액 (M)", f"${profile.get('marketCapitalization', 0):,.0f}")

            st.divider()
            st.subheader("🤖 AI 전략 분석")

            # [✨ FINAL] 두 버튼을 'AI 종합 분석' 단일 버튼으로 통합
            if st.button("💡 AI 종합 분석", use_container_width=True, type="primary"):
                # 이전 분석 결과 관련 세션 상태 초기화
                st.session_state.changelog_for_display = ""
                st.session_state.last_analysis_text = "" 

                analysis_logs = load_analysis_log(main_ticker)

                # --- 분기 로직: 기록 유무에 따라 지능형 분석 또는 전체 분석 수행 ---
                if not analysis_logs.empty:
                    # [CASE 1: 기록 있음 -> 지능형 분석 수행]
                    with st.spinner("1/3) 종합 팩터 시트 생성 중..."):
                        latest_log = analysis_logs.iloc[0]
                        previous_analysis_text = latest_log.get('전체 분석 내용', '')
                        candles_with_indicators = add_technical_indicators(candles_df.copy())
                        ratings = get_ratings_from_hub(main_ticker)
                        insider_trans = get_insider_trans_from_hub(main_ticker)
                        factor_sheet = create_factor_sheet(main_ticker, latest_log, candles_with_indicators, news, ratings, insider_trans)
                    
                    with st.spinner("2/3) 팩터 기반 보고서 업데이트 중..."):
                        evolved_report = "".join(list(stream_evolved_report(previous_analysis_text, factor_sheet, main_ticker)))
                        st.session_state.last_analysis_text = evolved_report
                        st.session_state.last_analysis_ticker = main_ticker

                    with st.spinner("3/3) AI가 변경 내역 브리핑을 생성 중..."):
                        changelog_header = f"💡 **AI 변경점 브리핑 (Changelog)**\n\n**[AI가 입력받은 종합 팩터 시트]**\n```\n{factor_sheet}\n```\n---"
                        ai_briefing = generate_changelog(previous_analysis_text, evolved_report, factor_sheet)
                        st.session_state.changelog_for_display = changelog_header + "\n\n" + ai_briefing
                
                else:
                    # [CASE 2: 기록 없음 -> 첫 전체 분석 수행]
                    st.info("첫 분석을 시작합니다. 전체 분석을 수행합니다...")
                    with st.spinner("AI가 최신 정보로 전체 분석을 수행 중입니다..."):
                        # '참고 자료'는 제거하고 '핵심 지침'만 전달
                        core_principle_files = synchronize_knowledge_files()
                        
                        # 전체 분석에 필요한 모든 데이터 준비 (기존 '전체 재분석' 로직과 동일)
                        portfolio_df, _, cash_df = load_data_from_gsheet()
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
                        support_levels = calculate_support_levels(candles_df)
                        dynamic_trends = calculate_dynamic_trends(candles_df.copy(), quote)
                        market_context = get_market_status_data()

                        # AI 호출 (reference_files 인자 없이 호출)
                        full_analysis_text = "".join(list(stream_and_capture_analysis(main_ticker, profile, quote, financials_df, tech_summary, news, full_context, support_levels, dynamic_trends, market_context, core_principle_files)))
                        st.session_state.last_analysis_text = full_analysis_text
                        st.session_state.last_analysis_ticker = main_ticker

                st.rerun()

            st.markdown("---")

            # ✨ FINAL REVISED: 통합된 '최종 분석 결과' 표시 로직
            # 지능형 재분석 결과 브리핑을 먼저 표시
            if "changelog_for_display" in st.session_state and st.session_state.changelog_for_display:
                with st.expander("💡 AI 변경점 브리핑", expanded=True):
                    st.markdown(st.session_state.changelog_for_display)
            
            # 항상 '공식적인 최종 분석 결과'를 표시 (지능형/전체 분석 모두 여기에 반영됨)
            if st.session_state.get("last_analysis_text") and st.session_state.get("last_analysis_ticker") == main_ticker:
                st.subheader("💡 AI 종합 분석 보고서")
                st.markdown(st.session_state.last_analysis_text)
                
                # 저장 버튼 로직은 이제 항상 정확하게 작동
                if st.button("💾 현재 분석 결과 저장", key=f"gemini_save_{main_ticker}"):
                    with st.spinner("분석 결과를 Google Sheets에 저장하는 중..."):
                        tech_summary_save = generate_technical_summary(add_technical_indicators(candles_df.copy()))
                        support_levels_save = calculate_support_levels(candles_df)
                        data_snapshot = {
                            "profile": profile, "quote": quote,
                            "news_headlines": [item['headline'] for item in news[:5]] if news else [],
                            "tech_summary": tech_summary_save, "support_levels": support_levels_save
                        }
                        
                        analysis_text_to_save = st.session_state.get("last_analysis_text", "")
                        try:
                            summary_text = analysis_text_to_save.split("#### 4.")[0].strip().replace("*", "").replace("#", "")[-200:] + "..."
                        except Exception:
                            summary_text = analysis_text_to_save.strip().replace("*","").replace("#","")[:200] + "..."
                        
                        log_entry = {
                            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "종목코드": main_ticker,
                            "AI_Model": st.session_state.get("last_model_used", "N/A"),
                            "당시 주가": quote.get('c', 0) if quote else 0,
                            "분석 요약": summary_text,
                            "전체 분석 내용": analysis_text_to_save,
                            "주요 데이터": json.dumps(data_snapshot, ensure_ascii=False, indent=2)
                        }

                        if save_analysis_to_gsheet(log_entry):
                            st.toast("✅ 분석 결과를 성공적으로 저장했습니다!")
                            st.cache_data.clear() # 캐시 클리어 후 재실행하여 최신 로그 반영
                            st.rerun()
                        else:
                            st.error("분석 결과 저장에 실패했습니다.")
            else:
                st.info(f"'{main_ticker}'에 대한 분석 기록이 없습니다. '재분석 실행' 버튼을 눌러 새 분석을 시작하세요.")
        
        # 이하 다른 탭들(tech_tab, fin_tab, news_tab, log_tab)의 로직은 기존 코드를 유지합니다.
        # [중요] 단, 각 탭에서 parsed_data를 참조하기 전에, st.session_state.last_analysis_text를 파싱하는 로직이 필요합니다.
        
        # '단일 진실 공급원' 파싱 로직 (각 탭 렌더링 직전에 위치)
        parsed_data = {}
        if st.session_state.get("last_analysis_text"):
            parsed_data = structure_recommendation(st.session_state.last_analysis_text)
       

        with tech_tab:
            st.subheader("📈 기술적 분석")

            if parsed_data and parsed_data.get('technical_briefing'):
                st.info(parsed_data['technical_briefing'])

            if not candles_df.empty and len(candles_df) > 60:
                candles_df_tech = add_technical_indicators(candles_df.copy())
                st.divider()
                st.subheader("AI 추천 매매 구간 시각화")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Candlestick & AI Zones', 'MACD', 'RSI'), row_heights=[0.6, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=candles_df_tech['Date'], open=candles_df_tech['Open'], high=candles_df_tech['High'], low=candles_df_tech['Low'], close=candles_df_tech['Close'], name='Price'), row=1, col=1)

                if parsed_data:
                    buy_zones = parsed_data.get('buy_zones')
                    sell_zones = parsed_data.get('sell_zones')
                    if buy_zones and buy_zones.get('zone1'):
                        fig.add_hrect(y0=buy_zones['zone1'][0], y1=buy_zones['zone1'][1], line_width=0, fillcolor="green", opacity=0.2, annotation_text="1st Buy Zone", annotation_position="bottom right", row=1, col=1)
                    if sell_zones and sell_zones.get('zone1'):
                        fig.add_hrect(y0=sell_zones['zone1'][0], y1=sell_zones['zone1'][1], line_width=0, fillcolor="red", opacity=0.2, annotation_text="1st Sell Zone", annotation_position="bottom right", row=1, col=1)
                
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SMA20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SMA60'], name='SMA 60', line=dict(color='purple', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SignalLine'], name='Signal Line', line=dict(color='red', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['RSI14'], name='RSI 14', line=dict(color='royalblue', width=1)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.update_layout(height=800, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else: 
                st.info(f"'{main_ticker}'에 대한 차트 데이터가 부족하여 매수 구간을 계산할 수 없습니다.")
            
            
        with fin_tab:
            st.subheader("💰 펀더멘털 분석")
            # [신규] AI 펀더멘털 브리핑 표시
            if parsed_data.get('fundamental_briefing'):
                st.info(parsed_data['fundamental_briefing'])

            st.divider()

            st.subheader("핵심 재무 지표 (연간)")
            if not financials_df.empty: st.dataframe(financials_df.style.format("{:,.2f}", na_rep="-"))
            else: st.warning(f"'{main_ticker}'에 대한 재무 데이터를 찾을 수 없습니다.")
            st.subheader("분기별 실적 발표 내역")
            if not earnings_data.empty:
                format_dict = {'실제 EPS': '{:.2f}', '예상 EPS': '{:.2f}', 'EPS 서프라이즈 (%)': '{:.2f}%'}
                st.dataframe(earnings_data.style.format(format_dict, na_rep="-"))
                if 'EPS 서프라이즈 (%)' in earnings_data.columns:
                    fig_earn = px.bar(earnings_data, x='발표 분기', y='EPS 서프라이즈 (%)', color='EPS 결과', color_discrete_map={'Beat': 'green', 'Miss': 'red', 'Meet': 'blue'})
                    st.plotly_chart(fig_earn, use_container_width=True)
            else: st.warning(f"'{main_ticker}'에 대한 실적 내역이 없습니다.")
            st.subheader("경쟁사 비교")
            if peers:
                peer_df = get_peer_summary([p for p in peers if p != main_ticker][:5])
                if not peer_df.empty: st.dataframe(peer_df.set_index('Ticker').style.format({"Market Cap (M)": "{:,.0f}", "% Change": "{:.2f}%"}, na_rep="-").background_gradient(cmap='RdYlGn', subset=['% Change']))
                else: st.info("경쟁사 정보를 가져올 수 없습니다.")
            else: st.info("경쟁사 정보가 없습니다.")
            
        with news_tab:
            st.subheader("📰 뉴스 및 기업 개요")
            # [신규] AI 뉴스 브리핑 표시
            if parsed_data.get('news_briefing'):
                st.info(parsed_data['news_briefing'])

            st.divider()
            if profile:
                st.subheader(f"기업 프로필: {profile.get('name', main_ticker)}")
                col1, col2 = st.columns([1, 4]); col1.image(profile.get('logo'), width=100)
                with col2: st.text(f"Industry: {profile.get('finnhubIndustry')}"); st.link_button("Visit Website", profile.get('weburl'))
                if next_earnings_date: st.info(f"**다음 실적 발표 예정일:** {next_earnings_date}")
            else: st.warning("기업 프로필 정보가 없습니다.")
            st.divider()
            st.subheader("최신 관련 뉴스")
            if news:
                for item in news[:10]:
                    news_date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M')
                    st.markdown(f"**[{item['headline']}]({item['url']})**\n- *Source: {item['source']} | {news_date}*")
            else: st.info("관련 뉴스가 없습니다.")

        with log_tab:
            st.subheader("📜 과거 분석 기록 보관소")
            analysis_logs = load_analysis_log(main_ticker)
            if not analysis_logs.empty:
                for index, row in analysis_logs.iterrows():
                    log_time = pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')
                    with st.expander(f"**{log_time}** | 당시 주가: ${float(row.get('당시 주가', 0)):.2f}"):
                        st.markdown(row.get('전체 분석 내용', '저장된 전체 내용이 없습니다.'))
            else:
                st.info(f"'{main_ticker}'에 대한 과거 분석 기록이 없습니다.")

    else:
        st.info("사이드바에서 분석할 Ticker를 입력하고 '분석 실행' 버튼을 클릭하여 상세 분석을 시작하세요.")


        
with st.sidebar:
    st.header("Controls")
    view_options = ["🔭 시장 건강 상태", "💼 포트폴리오", "📡 레이더", "📡 탐색", "🔍 상세 분석"] #<-- 이렇게 변경
    
    # st.radio의 현재 선택값을 selected_view 변수에 저장
    selected_view = st.radio(
        "Select View", 
        view_options, 
        index=view_options.index(st.session_state.active_view), 
        #horizontal=True,
        key="view_selector"
    )
    
    if selected_view != st.session_state.active_view:
        st.session_state.active_view = selected_view
        st.rerun()

    st.divider()
    
    default_tickers = ""
    if not st.session_state.watchlist_df.empty and '종목코드' in st.session_state.watchlist_df.columns:
        default_tickers = ", ".join(st.session_state.watchlist_df['종목코드'].dropna().unique().tolist())
    
    tickers_input = st.text_area("Ticker(s) for Analysis", value=default_tickers, help="분석할 종목의 Ticker를 쉼표(,)로 구분하여 입력하세요.")
    
    if st.button("🔍 분석 실행", use_container_width=True, type="primary"):
        # --- ✨ NEW: 새로운 분석 시작 시, 이전 분석 기록 초기화 ---
        # 이전에 남아있을 수 있는 모든 분석 결과 관련 세션 상태를 깨끗하게 비웁니다.
        keys_to_clear = [
            "last_analysis_text", 
            "last_analysis_ticker", 
            "changelog_for_display",
            "beta_analysis_result", # 혹시 모를 이전 베타 결과도 함께 제거
            "beta_changelog",
            "beta_evolved_report"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # -----------------------------------------------------------

        st.session_state.analysis_tickers = [ticker.strip().upper() for ticker in tickers_input.replace(',', '\n').split('\n') if ticker.strip()]
        st.session_state.active_view = "🔍 상세 분석"
        # st.session_state.last_analysis_text = None # 위에서 del로 대체되었으므로 주석 처리 또는 삭제
        st.session_state.last_saved_ticker = None
        st.rerun()

    st.divider()
    st.info("포트폴리오, 현금, 관심종목은 Google Sheets에서 직접 수정해주세요.")
    if st.button("🔄 Reload Data & Clear Cache", use_container_width=True):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

