import streamlit as st
from pathlib import Path
import joblib
import numpy as np

# Streamlit Cloud cwd = 레포 루트, app은 src/app.py → pkl은 src/ 또는 루트에 두면 됨
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MODEL_NAMES = ("nb_model.pkl", "tfidf_vectorizer.pkl")


def _resolve_pkl(name: str) -> Path:
    for base in (_HERE, _ROOT):
        p = base / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"'{name}' 을(를) 찾을 수 없습니다. 다음 중 한 곳에 넣어 주세요: "
        f"{_HERE} 또는 {_ROOT} (Git에 커밋되어 있어야 배포 환경에서 읽힙니다)"
    )


# --- 1. 저장된 모델 및 벡터라이저 불러오기 ---
@st.cache_resource
def load_models():
    model = joblib.load(_resolve_pkl(_MODEL_NAMES[0]))
    vectorizer = joblib.load(_resolve_pkl(_MODEL_NAMES[1]))
    return model, vectorizer


st.set_page_config(page_title="AI 탐지기 시연", layout="centered")
model, vectorizer = load_models()

# --- 2. UI 구성 ---
st.title("실시간 AI 생성 텍스트 탐지 시스템")
st.write("AI 생성 문장 탐지 최종 모델 (나이브 베이즈) 기반 탐지기입니다.")

# 텍스트 입력창
user_input = st.text_area("분석할 텍스트를 입력하세요", placeholder="내용을 입력하고 Ctrl+Enter를 누르세요.", height=200)

if st.button("AI 판별 시작"):
    if user_input:
        # --- 3. 실시간 예측 프로세스 ---
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)
        probabilities = model.predict_proba(input_tfidf)

        st.divider()
        if prediction[0] == 1:
            st.error("분석 결과: AI 생성 문장일 가능성이 매우 높습니다")
            st.write(f"확률: {probabilities[0][1]*100:.2f}%")
        else:
            st.success("분석 결과: 사람이 작성한 문장일 가능성이 매우 높습니다")
            st.write(f"확률: {probabilities[0][0]*100:.2f}%")

        st.progress(float(probabilities[0][1]))
    else:
        st.warning("텍스트를 입력해주세요.")
