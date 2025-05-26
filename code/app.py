import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ✅ 모델 로드
@st.cache_resource
def load_my_model():
    return load_model('realorai_model.h5')

model = load_my_model()

# ✅ 이미지 전처리
def preprocess_uploaded_image(img, size=(128, 128)):
    img = img.convert("RGB").resize(size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ✅ 사이드바 메뉴
menu = st.sidebar.selectbox("📂 기능 선택", ["명화 vs AI 그림 분류기", "모델 성능 시각화"])

# -------------------------------------------------------------------------------------
# 🎨 기능 1: 그림 판별기
# -------------------------------------------------------------------------------------
if menu == "명화 vs AI 그림 분류기":
    st.title("🎨 명화 vs AI 그림 분류기")
    st.write("업로드한 이미지를 AI가 명화인지 AI 그림인지 판별해줍니다.")

    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)

        img_array = preprocess_uploaded_image(image)
        prediction = model.predict(img_array)
        prob = prediction[0][0]

        st.subheader("🔍 판별 결과")
        st.write(f"AI 그림일 확률: **{prob:.2%}**")

        if prob >= 0.5:
            st.success("이 이미지는 **AI 그림**으로 판별되었습니다.")
        else:
            st.success("이 이미지는 **명화(사람이 그린 그림)**으로 판별되었습니다.")

# -------------------------------------------------------------------------------------
# 📊 기능 2: 모델 성능 시각화
# -------------------------------------------------------------------------------------
elif menu == "모델 성능 시각화":
    st.title("📊 모델 성능 시각화")

    # ✅ 학습 결과 불러오기
    acc = np.load('history_acc.npy')
    val_acc = np.load('history_val_acc.npy')
    loss = np.load('history_loss.npy')
    val_loss = np.load('history_val_loss.npy')

    st.subheader("📈 정확도 그래프")
    plt.figure(figsize=(8, 4))
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    st.pyplot(plt)

    st.subheader("📉 손실 그래프")
    plt.figure(figsize=(8, 4))
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    st.pyplot(plt)

    # ✅ 테스트셋 불러오기
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()

    st.subheader("🧮 혼동 행렬")
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)

    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred_binary)
    st.text(report)