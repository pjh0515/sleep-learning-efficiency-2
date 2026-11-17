
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="수면 시간과 학습 효율 분석",
    page_icon="📚",
    layout="wide"
)

st.title("📚 수면 시간과 학습 효율 분석")
st.write("수면 시간과 학습 습관을 바탕으로 학습 효율을 예측하는 웹앱입니다.")

# 1. 연습용 가상 데이터 만들기
@st.cache_data
def make_data():
    np.random.seed(42)
    n = 200

    sleep = np.random.uniform(4, 10, n)
    study = np.random.uniform(1, 8, n)
    satisfaction = np.random.randint(1, 6, n)
    fatigue = np.random.randint(1, 6, n)

    # 가상의 규칙으로 학습 효율 등급 생성
    score = (
        sleep * 0.35
        + study * 0.25
        + satisfaction * 0.7
        - fatigue * 0.5
        + np.random.normal(0, 1, n)
    )

    efficiency = np.where(score >= 4.5, "높음", "낮음")

    return pd.DataFrame({
        "수면 시간": sleep,
        "공부 시간": study,
        "수면 만족도": satisfaction,
        "피로도": fatigue,
        "학습 효율": efficiency
    })

df = make_data()

# 2. 데이터 확인
st.subheader("1. 연습용 데이터 확인")
st.caption("아래 데이터는 실제 조사 자료가 아닌 가상 데이터입니다.")
st.dataframe(df.head(10), use_container_width=True)

# 3. 머신러닝 모델 학습
X = df[["수면 시간", "공부 시간", "수면 만족도", "피로도"]]
y = df["학습 효율"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 4. 모델 성능 표시
st.subheader("2. 인공지능 모델 성능")
st.metric("정확도 (Accuracy)", f"{accuracy * 100:.1f}%")
st.caption("가상 데이터로 학습하고 평가한 결과입니다.")

# 5. 사용자 입력 및 예측
st.subheader("3. 학습 효율 예측하기")

col1, col2 = st.columns(2)

with col1:
    sleep_input = st.slider("수면 시간", 3.0, 12.0, 7.0, 0.5)
    study_input = st.slider("하루 공부 시간", 0.0, 12.0, 3.0, 0.5)

with col2:
    satisfaction_input = st.slider("수면 만족도", 1, 5, 3)
    fatigue_input = st.slider("피로도", 1, 5, 3)

if st.button("학습 효율 예측하기"):
    new_data = pd.DataFrame({
        "수면 시간": [sleep_input],
        "공부 시간": [study_input],
        "수면 만족도": [satisfaction_input],
        "피로도": [fatigue_input]
    })

    result = model.predict(new_data)[0]
    probability = model.predict_proba(new_data).max()

    st.success(f"예측된 학습 효율: {result}")
    st.write(f"모델의 해당 예측 확률: {probability * 100:.1f}%")

# 6. 수면 시간과 학습 효율 데이터 그래프
st.subheader("4. 수면 시간과 학습 효율의 관계")

chart_data = df.groupby("학습 효율")["수면 시간"].mean()
st.bar_chart(chart_data)

st.info(
    "이 웹앱은 작동 방법을 확인하기 위한 시제품입니다. "
    "실제 연구에는 익명화된 설문 자료와 적절한 분석이 필요합니다."
)
