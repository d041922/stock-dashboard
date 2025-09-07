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
        reference_files = [f for f in final_gemini_files if f.display_name not in core_file_names]
        
        return core_principle_files, reference_files

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
def stream_and_capture_analysis(ticker, profile, quote, financials_df, tech_summary, news, portfolio_context, support_levels, dynamic_trends, market_context, core_principle_files, reference_files):
    model_name = 'gemini-2.5-pro'
    
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
    ref_files_names = ", ".join([f.display_name for f in reference_files]) if reference_files else "없음"
    
    master_prompt = f"""
    **SYSTEM ROLE:** 당신은 월스트리트 최고의 금융 분석가이자, 'MASTER'라는 투자자를 보좌하는 AI 전략 파트너입니다.

**[최상위 지침: MASTER의 투자 철학 (절대적 기준)]**
    - **첨부된 파일 중 "{core_files_names}"** 의 내용은 당신의 모든 분석과 추천을 위한 최상위 원칙입니다.
    - 이 지침은 다른 어떤 데이터나 참고 자료보다 항상 우선합니다.

    **[참고 자료 (분석의 깊이를 더하기 위한 보조 정보)]**
    - 당신의 분석 품질을 높이기 위해 **첨부된 파일 중 "{ref_files_names}"** 를 참고할 수 있습니다.
    - 만약 참고 자료의 내용이 최상위 지침과 충돌할 경우, 반드시 최상위 지침을 따라야 합니다.

    **분석 대상:** {ticker}
    **MASTER의 현재 포트폴리오 상황:**
    {portfolio_context}

    **입력 데이터:**
    1. 기업 개요: {profile.get('name', 'N/A')}, 산업: {profile.get('finnhubIndustry', 'N/A')}
    2. 현재 시세: 현재가 ${quote.get('c', 0):.2f}
    3. 핵심 재무 요약: \n{financials_df.tail(3).to_string() if not financials_df.empty else "N/A"}
    4. 기술적 분석 요약: \n- {"\n- ".join(tech_summary)}
    5. 최신 뉴스 요약: \n- {"\n- ".join([item['headline'] for item in news[:5]]) if news else "N/A"}
    6. 시스템이 계산한 잠재적 기술적 지지선 리스트: {support_levels}
    7. 최근 주가 동향: {trends_text}
    8. 현재 거시 경제 상황: {market_context_text}

    **MISSION:**

    **[중요] 분석을 시작하기 전에, 다음 형식에 맞춰 당신이 참고한 지식 파일의 전체 목록을 가장 먼저 출력해야 한다:**
    "---
    **[Knowledge Core 연동 확인]**
    - **최상위 지침 파일:** [파일 A, 파일 B, ...]
    - **참고 자료 파일:** [파일 C, 파일 D, ...]
    ---"


    **첨부된 "{core_files_names}"와 "{ref_files_names}"를 반드시 참고하되 최상위 지침을 기준으로**, 모든 입력 데이터를 종합하여 아래 4가지 질문에 답하는 상세한 분석 보고서를 생성하십시오.
    특히 "4. 어떻게 행동해야 하는가?" 항목을 작성할 때, 최종적으로 결정한 추천 매수 구간과 판단 근거를 다음과 같은 **데이터 블록(Data Block)** 형식으로 텍스트 안에 반드시 포함시켜야 합니다.

    ```text
    [BUY_ZONES]
    zone1_start: 155.0
    zone1_end: 160.0
    zone2_start: 130.0
    zone2_end: 140.0
    rationale: 기술적 지지선은 $150에 있지만, 최근 긍정적인 뉴스를 반영하여 매수 구간을 상향 조정함.
    [/BUY_ZONES]
    ```

    만약 매수를 추천하지 않는다면, 모든 zone 값에 'N/A'를 입력하십시오.

    ---
    ### 💎 {ticker} 전략 브리핑
    *Analysis Model: `{model_name}`*
    #### 1. 좋은 종목인가? (What to Buy?)
    ... (자유롭게 서술)
    #### 2. 좋은 시기인가? (When to Buy?)
    ... (자유롭게 서술)
    #### 3. 좋은 가격인가? (What Price?)
    ... (자유롭게 서술)
    #### 4. 어떻게 행동해야 하는가? (How to Act?)
    ... (위의 규칙에 따라 데이터 블록을 포함하여 서술)
    """

    full_response = []
    try:
        # [핵심 수정] generate_content 호출 시, 프롬프트와 파일 객체 리스트를 함께 전달
        all_files = core_principle_files + reference_files
        response = model.generate_content([master_prompt] + all_files, stream=True)

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

def structure_recommendation(full_analysis_text):
    """AI가 생성한 텍스트에서 [BUY_ZONES] 데이터 블록을 추출하고 파싱합니다."""
    try:
        # 데이터 블록 추출
        match = re.search(r"\[BUY_ZONES\](.*?)\[/BUY_ZONES\]", full_analysis_text, re.DOTALL)
        if not match:
            return None
        
        content = match.group(1).strip()
        
        # 각 라인을 파싱하여 딕셔너리로 변환
        reco = {}
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                reco[key.strip()] = value.strip()

        # 숫자 변환 및 최종 데이터 구조화
        def to_float_or_none(val):
            try: return float(val)
            except (ValueError, TypeError): return None

        zone1_start = to_float_or_none(reco.get('zone1_start'))
        zone1_end = to_float_or_none(reco.get('zone1_end'))
        zone2_start = to_float_or_none(reco.get('zone2_start'))
        zone2_end = to_float_or_none(reco.get('zone2_end'))
        
        structured_data = {
            'buy_zone_1': (zone1_start, zone1_end) if zone1_start and zone1_end else None,
            'buy_zone_2': (zone2_start, zone2_end) if zone2_start and zone2_end else None,
            'rationale': reco.get('rationale', '판단 근거를 추출하지 못했습니다.')
        }
        return structured_data
    except Exception:
        return None

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
            profile, quote = get_company_profile(ticker), get_quote(ticker)
            summary_data.append({"Ticker": ticker, "Name": profile.get('name', ticker), "Market Cap (M)": profile.get('marketCapitalization', 0), "% Change": quote.get('dp', 0)})
        except: continue
    return pd.DataFrame(summary_data)

@st.cache_data(ttl=300) # 5분마다 데이터 갱신
def get_market_status_data():
    """주요 시장 지수 및 거시 경제 지표 데이터를 가져옵니다."""
    data = {}
    try:
        # 주요 지수 Ticker
        tickers = {
            "S&P 500": "^GSPC",
            "Nasdaq": "^IXIC",
            "KOSPI": "^KS11",
            "VIX": "^VIX",
            "US 10Y": "^TNX"
        }
        
        market_data = yf.download(list(tickers.values()), period="5d", progress=False)
        
        for name, ticker in tickers.items():
            price = market_data['Close'][ticker].iloc[-1]
            change = market_data['Close'][ticker].iloc[-1] - market_data['Close'][ticker].iloc[-2]
            change_percent = (change / market_data['Close'][ticker].iloc[-2]) * 100
            data[name] = {"price": price, "change": change, "change_percent": change_percent}

        # 경제 뉴스
        data['news'] = finnhub_client.general_news('general', min_id=0)

        return data
    except Exception as e:
        st.error(f"시장 데이터 로딩 중 오류 발생: {e}")
        return None


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

# --- [추가] 레이더 뷰 ---
elif st.session_state.active_view == "📡 레이더":
    st.header("📡 Stock Radar")
    
    watchlist_df = st.session_state.watchlist_df
    
    if watchlist_df.empty or '종목코드' not in watchlist_df.columns:
        st.info("관심 종목이 없습니다. Google Sheets의 'Watchlist'에 종목을 추가해주세요.")
    else:
        tickers = watchlist_df['종목코드'].dropna().unique().tolist()
        
        @st.cache_data(ttl=600)
        def get_radar_data(ticker_list):
            data = yf.download(ticker_list, period="1y", progress=False)
            summary_list = []
            for ticker in ticker_list:
                try:
                    # MultiIndex에서 단일 종목 데이터 추출
                    if len(ticker_list) > 1:
                        hist = data.loc[:, (slice(None), ticker)]
                        # MultiIndex 제거
                        hist.columns = hist.columns.droplevel(1)
                    else:
                        hist = data
                    
                    if hist.empty: continue
                    
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                    high_52w = hist['High'].max()
                    mdd_percent = ((current_price - high_52w) / high_52w) * 100
                    
                    delta = hist['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs.iloc[-1]))
                    
                    volume_change = (hist['Volume'].iloc[-1] / hist['Volume'].rolling(window=20).mean().iloc[-1]) * 100

                    summary_list.append({
                        "종목코드": ticker,
                        "현재가": current_price,
                        "등락률(%)": change_percent,
                        "고점대비(%)": mdd_percent,
                        "RSI": rsi,
                        "거래량(%)": volume_change
                    })
                except Exception:
                    continue
# [수정] 스타일링을 위해 데이터 타입을 명시적으로 숫자로 변환
            if not summary_list:
                return pd.DataFrame()

            df = pd.DataFrame(summary_list)
            numeric_cols = ["현재가", "등락률(%)", "고점대비(%)", "RSI", "거래량(%)"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        with st.spinner("레이더 데이터를 스캔하는 중..."):
            radar_df = get_radar_data(tickers)

        if not radar_df.empty:
            
            
            # [수정] 포맷터를 먼저 정의하여 오류 해결
            formatter = {
                "등락률(%)": "{:,.2f}%",
                "고점대비(%)": "{:,.2f}%",
                "RSI": "{:.1f}",
                "거래량(%)": "{:,.0f}%",
                # [개선] 종목별로 통화 단위를 올바르게 적용
                "현재가": lambda x: f"₩{x:,.0f}" if ".KS" in st.session_state.get('ticker_info', {}).get(x, '') or ".KQ" in st.session_state.get('ticker_info', {}).get(x, '') else f"${x:,.2f}"
            }
            # 종목코드별 통화 정보 저장을 위한 임시 상태 저장
            temp_ticker_info = {}
            for index, row in radar_df.iterrows():
                temp_ticker_info[row['현재가']] = row['종목코드']
            st.session_state.ticker_info = temp_ticker_info

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


# --- [수정] 시장 현황 뷰 ---
elif st.session_state.active_view == "📈 시장 현황":
    st.header("📈 Market Dashboard")
    
    market_data = get_market_status_data()

    if market_data:
        st.subheader("주요 시장 지수")
        cols = st.columns(5)
        indices = ["S&P 500", "Nasdaq", "KOSPI", "VIX", "US 10Y"]
        for i, name in enumerate(indices):
            if name in market_data:
                d = market_data[name]
                # VIX와 금리는 %가 아니므로 delta 포맷을 다르게 적용
                if name in ["VIX", "US 10Y"]:
                     cols[i].metric(
                        label=name,
                        value=f"{d['price']:.2f}",
                        delta=f"{d['change']:.2f}"
                    )
                else:
                    cols[i].metric(
                        label=name,
                        value=f"{d['price']:,.2f}",
                        delta=f"{d['change']:,.2f} ({d['change_percent']:.2f}%)"
                    )
        
        st.divider()
        
        st.subheader("주요 경제 뉴스")
        if market_data.get('news'):
            for item in market_data['news'][:5]:
                news_date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
                st.markdown(f"**[{item['headline']}]({item['url']})** - *{news_date}, {item['source']}*")
        else:
            st.warning("주요 경제 뉴스를 불러올 수 없습니다.")
    else:
        st.error("시장 현황 데이터를 가져오는 데 실패했습니다.")



# --- 상세 분석 뷰 ---
elif st.session_state.active_view == "🔍 상세 분석":
    if 'analysis_tickers' in st.session_state and st.session_state.analysis_tickers:
        main_ticker = st.session_state.analysis_tickers[0]
        st.header(f"🔍 {main_ticker} 상세 분석")
        
        with st.spinner(f"'{main_ticker}' 상세 데이터를 가져오는 중..."):
            profile, quote, news, financials_df, peers, earnings_data, next_earnings_date, candles_df = (get_company_profile(main_ticker), get_quote(main_ticker), get_company_news(main_ticker), get_basic_financials(main_ticker), get_company_peers(main_ticker), get_company_earnings(main_ticker), get_earnings_calendar(main_ticker), get_stock_candles(main_ticker))
        
        # [수정] 탭 구조를 5개로 재구성 (과거 분석 기록 탭 분리)
        analysis_tab_names = ["💎 종합 진단", "📜 과거 분석 기록", "📈 기술적 분석", "💰 펀더멘털", "📰 뉴스 및 개요"]
        diag_tab, log_tab, tech_tab, fin_tab, news_tab = st.tabs(analysis_tab_names)

        with diag_tab:
            st.subheader(f"💎 {main_ticker} 종합 진단")

            # --- 1. 핵심 지표 요약 ---
            cols = st.columns(4)
            if quote and quote.get('c') != 0:
                cols[0].metric("현재가", f"${quote.get('c', 0):.2f}", f"{quote.get('d', 0):.2f}$ ({quote.get('dp', 0):.2f}%)")
            candles_df_tech_diag = add_technical_indicators(candles_df.copy())
            if not candles_df_tech_diag.empty and 'RSI14' in candles_df_tech_diag.columns and not pd.isna(candles_df_tech_diag['RSI14'].iloc[-1]):
                latest_rsi = candles_df_tech_diag['RSI14'].iloc[-1]
                cols[1].metric("RSI (14일)", f"{latest_rsi:.2f}")
                high_52w = candles_df['High'].max()
                if high_52w > 0:
                    mdd_percent = ((quote.get('c', 0) - high_52w) / high_52w) * 100
                    cols[2].metric("52주 고점 대비", f"{mdd_percent:.2f}%")
            if profile:
                cols[3].metric("시가총액 (M)", f"${profile.get('marketCapitalization', 0):,.0f}")
            st.divider()

# --- 2. AI 최종 권고 ---
            st.subheader("🤖 AI 최종 권고")
            
            if st.button("🚀 GEMINI 전략 분석 실행", use_container_width=True, type="primary", key=f"gemini_run_{main_ticker}"):
                # 1단계: 지식 파일 자동 동기화
                core_principle_files, reference_files = synchronize_knowledge_files()
                
                # 2단계: 컨텍스트 준비 및 AI 분석 실행
                with st.spinner("AI가 분석 중입니다..."):
                    st.session_state.last_analysis_text = None
                    st.session_state.structured_reco = None
                    st.session_state.last_saved_ticker = None
                    
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

                # 2단계: AI 분석가 실행
                with st.spinner("AI가 분석 중입니다 (1/2)..."):
                    analysis_chunks = list(stream_and_capture_analysis(
                        main_ticker, profile, quote, financials_df, tech_summary, news, 
                        full_context, support_levels, dynamic_trends, market_context, core_principle_files, reference_files
                    ))
                    final_analysis = "".join(analysis_chunks)
                    st.session_state.last_analysis_text = final_analysis
                    st.session_state.last_analysis_ticker = main_ticker

                # 3단계: AI 전략가 실행
                with st.spinner("AI가 권고안을 구조화 중입니다 (2/2)..."):
                    structured_reco = structure_recommendation(final_analysis)
                    st.session_state.structured_reco = structured_reco
                st.rerun()


            # [최종 수정] 구조화된 요약(액션 카드)과 서술형 원본을 모두 표시
            if st.session_state.get("structured_reco") and st.session_state.get("last_analysis_ticker") == main_ticker:
                reco = st.session_state.structured_reco
                
                # --- 핵심 액션 플랜 (요약) ---
                col1, col2 = st.columns(2)
                zone1 = reco.get('buy_zone_1')
                zone2 = reco.get('buy_zone_2')
                col1.metric("1차 추천 매수 구간 (AI)", f"${zone1[0]:.2f} ~ ${zone1[1]:.2f}" if zone1 else "N/A")
                col2.metric("2차 추천 매수 구간 (AI)", f"${zone2[0]:.2f} ~ ${zone2[1]:.2f}" if zone2 else "N/A")
                
                st.markdown("**💡 AI 판단 근거 요약**")
                st.info(reco.get('rationale', '판단 근거를 추출하지 못했습니다.'))
                st.divider()

                # --- 상세 권고안 (원본) ---
                st.markdown("**📖 AI 상세 권고안 원본**")
                try:
                    # 원본 텍스트에서 "How to Act?" 부분 전체를 다시 추출하여 표시
                    how_to_act_header = "#### 4. 어떻게 행동해야 하는가? (How to Act?)"
                    full_recommendation_text = st.session_state.last_analysis_text.split(how_to_act_header)[1].strip()
                    st.success(full_recommendation_text)
                except (IndexError, AttributeError):
                     st.error("상세 권고안 원본을 표시할 수 없습니다.")

            elif st.session_state.get("last_analysis_text"):
                st.warning("AI 권고안을 구조화하는데 실패했습니다. 아래 전체 분석 내용을 참고하십시오.")
            else:
                st.info("'GEMINI 전략 분석 실행' 버튼을 눌러 AI 분석을 수행해주세요.")

            
            with st.expander("🔍 전체 AI 분석 및 결과 저장"):
                if st.session_state.get("last_analysis_text") and st.session_state.get("last_analysis_ticker") == main_ticker:
                    st.markdown("---"); st.subheader("🤖 AI 전체 분석 내용 (원본)"); st.markdown(st.session_state.last_analysis_text); st.divider()
                    if "오류" not in st.session_state.last_analysis_text:
                        if st.session_state.get("last_saved_ticker") == main_ticker:
                            st.success("✅ 이 분석 결과는 '과거 분석 기록'에 저장되었습니다.")
                        else:
                            if st.button("💾 현재 분석 결과 저장", key=f"gemini_save_{main_ticker}"):
                                with st.spinner("분석 결과를 Google Sheets에 저장하는 중..."):
                                    analysis_text = st.session_state.get("last_analysis_text", "")
                                    try: summary_text = analysis_text.split("#### 4. 어떻게 행동해야 하는가? (How to Act?)")[1].strip().replace("*", "").replace("#", "")[:200] + "..."
                                    except (IndexError, AttributeError): summary_text = analysis_text.strip().replace("*","").replace("#","")[:200] + "..."
                                    log_entry = { "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "종목코드": main_ticker, "AI_Model": st.session_state.get("last_model_used", "N/A"), "당시 주가": quote.get('c', 0), "분석 요약": summary_text, "전체 분석 내용": analysis_text, "주요 데이터": json.dumps({}, ensure_ascii=False) }
                                    if save_analysis_to_gsheet(log_entry):
                                        st.session_state.last_saved_ticker = main_ticker; st.cache_data.clear(); st.rerun()
                                    else: st.error("분석 결과 저장에 실패했습니다.")

        with log_tab:
            st.subheader("📜 과거 분석 기록")
            analysis_logs = load_analysis_log(main_ticker)
            if not analysis_logs.empty:
                for index, row in analysis_logs.iterrows():
                    with st.expander(f"**{row['Timestamp']}** | 당시 주가: ${float(row.get('당시 주가', 0)):.2f} | 모델: {row.get('AI_Model', 'N/A')}"):
                        st.markdown(f"**요약:** {row.get('분석 요약', 'N/A')}")
                        st.markdown("---")
                        st.markdown(row.get('전체 분석 내용', '저장된 전체 내용이 없습니다.'))
            else:
                st.info(f"'{main_ticker}'에 대한 과거 분석 기록이 없습니다.")


        with tech_tab:
            st.subheader("📈 기술적 분석")
            if not candles_df.empty and len(candles_df) > 60:
                candles_df_tech = add_technical_indicators(candles_df.copy())
                st.subheader("기술적 분석 요약"); 
                tech_summary = generate_technical_summary(candles_df_tech)
                for point in tech_summary: st.markdown(f"- {point}")
                st.divider(); st.subheader("AI 추천 매수 구간 시각화")
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Candlestick & AI Buy Zones', 'MACD', 'RSI'), row_heights=[0.6, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=candles_df_tech['Date'], open=candles_df_tech['Open'], high=candles_df_tech['High'], low=candles_df_tech['Low'], close=candles_df_tech['Close'], name='Price'), row=1, col=1)
                
                if st.session_state.get("structured_reco") and st.session_state.get("last_analysis_ticker") == main_ticker:
                    reco = st.session_state.structured_reco
                    zone1 = reco.get('buy_zone_1'); zone2 = reco.get('buy_zone_2')
                    if zone1: fig.add_hrect(y0=zone1[0], y1=zone1[1], line_width=0, fillcolor="green", opacity=0.2, annotation_text="1st Buy Zone (AI)", annotation_position="bottom right", row=1, col=1)
                    if zone2: fig.add_hrect(y0=zone2[0], y1=zone2[1], line_width=0, fillcolor="red", opacity=0.2, annotation_text="2nd Buy Zone (AI)", annotation_position="bottom right", row=1, col=1)
                else: st.warning("종합 진단 탭에서 AI 분석을 먼저 실행해주세요.")
                
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SMA20'], name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SMA60'], name='SMA 60', line=dict(color='purple', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['SignalLine'], name='Signal Line', line=dict(color='red', width=1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=candles_df_tech['Date'], y=candles_df_tech['RSI14'], name='RSI 14', line=dict(color='royalblue', width=1)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.update_layout(height=800, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info(f"'{main_ticker}'에 대한 차트 데이터가 부족하여 매수 구간을 계산할 수 없습니다.")
            
        with fin_tab:
            st.subheader("💰 펀더멘털 분석")
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
    else:
        st.info("사이드바에서 분석할 Ticker를 입력하고 '분석 실행' 버튼을 클릭하여 상세 분석을 시작하세요.")
        
with st.sidebar:
    st.header("Controls")
    view_options = ["💼 포트폴리오", "📡 레이더", "📈 시장 현황", "🔍 상세 분석"]
    
    # st.radio의 현재 선택값을 selected_view 변수에 저장
    selected_view = st.radio(
        "Select View", 
        view_options, 
        index=view_options.index(st.session_state.active_view), 
        horizontal=True,
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
