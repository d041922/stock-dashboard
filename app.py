import streamlit as st
import pandas as pd
import plotly.express as px # plotly 라이브러리 추가

# --- 페이지 설정 ---
st.set_page_config(
    page_title="나의 주식 대시보드",
    page_icon="📈",
    layout="wide"
)

# --- 제목 ---
st.title("📈 나의 주식 대시보드")

# --- 엑셀 파일 업로드 ---
uploaded_file = st.file_uploader("여기에 포트폴리오 엑셀 파일을 올려주세요.", type=['xlsx'])

# 파일이 업로드 되었을 때만 아래 코드를 실행
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # --- 핵심 지표 계산 ---
    total_investment = df['총매입가'].sum()
    total_current_value = df['현재평가금액'].sum()
    total_profit_loss = total_current_value - total_investment
    
    if total_investment > 0:
        total_return_rate = (total_profit_loss / total_investment) * 100
    else:
        total_return_rate = 0

    # --- 핵심 지표 표시 ---
    st.subheader("요약")
    cols = st.columns(4)
    with cols[0]:
        st.metric(label="총 투자 원금", value=f"{total_investment:,.0f} 원")
    with cols[1]:
        st.metric(label="현재 총 자산", value=f"{total_current_value:,.0f} 원")
    with cols[2]:
        st.metric(label="총 평가 손익", value=f"{total_profit_loss:,.0f} 원")
    with cols[3]:
        st.metric(label="총 수익률", value=f"{total_return_rate:.2f} %")
    
    st.divider()

    # --- 데이터 시각화 (원형 차트) ---
    st.subheader("포트폴리오 비중")
    fig = px.pie(df, names='종목명', values='현재평가금액', title='자산 비중(%)')
    st.plotly_chart(fig) # plotly 차트를 streamlit에 표시
    
    st.divider() 
    
    # --- 상세 내역 표시 ---
    st.subheader("상세 내역")
    st.dataframe(df)