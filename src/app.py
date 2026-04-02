import streamlit as st
import joblib
import numpy as np

# --- 1. 저장된 모델 및 벡터라이저 불러오기 ---
@st.cache_resource
def load_models():
    model = joblib.load('nb_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# --- 2. UI 구성 ---
st.set_page_config(page_title="AI 탐지기 시연", layout="centered")
st.title("실시간 AI 생성 텍스트 탐지 시스템")
st.write("AI 생성 문장 탐지 최종 모델 (나이브 베이즈) 기반 탐지기입니다.")

# 텍스트 입력창
user_input = st.text_area("분석할 텍스트를 입력하세요", placeholder="내용을 입력하고 Ctrl+Enter를 누르세요.", height=200)

if st.button("AI 판별 시작"):
    if user_input:
        # --- 3. 실시간 예측 프로세스 ---
        # 사용자가 입력한 생 텍스트를 저장된 벡터라이저로 변환 (transform만 사용)
        input_tfidf = vectorizer.transform([user_input])
        
        # 예측 및 확률 계산
        prediction = model.predict(input_tfidf)
        probabilities = model.predict_proba(input_tfidf)
        
        # 결과 시각화
        st.divider()
        if prediction[0] == 1:
            st.error(f"분석 결과: AI 생성 문장일 가능성이 매우 높습니다")
            st.write(f"확률: {probabilities[0][1]*100:.2f}%")
        else:
            st.success(f"분석 결과: 사람이 작성한 문장일 가능성이 매우 높습니다")
            st.write(f"확률: {probabilities[0][0]*100:.2f}%")
            
        # 확률 바 표시
        st.progress(probabilities[0][1]) 
    else:
        st.warning("텍스트를 입력해주세요.")