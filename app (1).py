import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
IMG_SIZE = 128

# Load model
model = load_model("model/skin_model.h5")

# Load labels
df = pd.read_csv("dataset/metadata.csv")
le = LabelEncoder()
le.fit(df['dx'])
class_names = list(le.classes_)

# Disease info
disease_info = {
    'nv': "Nevus: Common mole, usually harmless.",
    'mel': "Melanoma: Dangerous skin cancer. Consult a doctor immediately.",
    'bcc': "Basal Cell Carcinoma: Slow-growing skin cancer.",
    'akiec': "Actinic Keratosis: Pre-cancerous lesion.",
    'bkl': "Benign Keratosis: Non-cancerous growth.",
    'df': "Dermatofibroma: Benign skin nodule.",
    'vasc': "Vascular lesion: Blood vessel related skin issue."
}

# ---------------- UI ----------------
st.set_page_config(page_title="Skin Disease Classifier", page_icon="🧴")

# 💜 Purple Theme (Fixed)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #2e1065, #4c1d95);
    color: #f3e8ff;
}

h1 {
    color: #ddd6fe;
    text-align: center;
}

/* Card */
.block-container {
    padding: 2rem;
    background-color: #3b0764;
    border-radius: 12px;
}

/* Upload text FIX */
section[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-weight: bold;
    font-size: 18px;
}

/* Upload box FIX */
section[data-testid="stFileUploader"] div {
    background-color: #ede9fe !important;
    color: #1e1b4b !important;
    border-radius: 10px;
}

/* Button */
.stButton>button {
    background-color: #7c3aed;
    color: white;
    border-radius: 8px;
}

/* Progress */
.stProgress > div > div > div > div {
    background-color: #a78bfa;
}
</style>
""", unsafe_allow_html=True)

st.title("🧴 Skin Disease Classifier")
st.write("Upload a skin image to predict the disease")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------------- PREPROCESS ----------------
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction)

    # 🎯 BIG CLEAR PREDICTION
    st.markdown(
        f"<h1 style='color:#e9d5ff;'>Prediction: {predicted_class.upper()}</h1>",
        unsafe_allow_html=True
    )

    # 📊 CLEAN CONFIDENCE BOX
    st.markdown(
        f"<div style='background-color:#ede9fe; padding:12px; border-radius:10px; color:#1e1b4b; font-weight:bold;'>📊 Confidence: {confidence*100:.2f}%</div>",
        unsafe_allow_html=True
    )

    # 📘 Disease Info Card
    if predicted_class in disease_info:
        st.markdown(
            f"<div style='background-color:#f5f3ff; padding:10px; border-radius:10px; color:#4c1d95;'>🧾 {disease_info[predicted_class]}</div>",
            unsafe_allow_html=True
        )

    # ⚠️ Suggestions
    if predicted_class == "mel":
        st.markdown("<p style='color:#fca5a5;'>⚠️ High Risk: Consult doctor immediately.</p>", unsafe_allow_html=True)
    elif predicted_class == "bcc":
        st.markdown("<p style='color:#fde68a;'>⚠️ Moderate Risk: Check recommended.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#bbf7d0;'>✅ Low Risk: Likely non-cancerous.</p>", unsafe_allow_html=True)

    # 📊 Bar Chart
    prob_df = pd.DataFrame({
        'Disease': class_names,
        'Probability': prediction[0]
    })

    st.subheader("📊 Prediction Distribution")
    st.bar_chart(prob_df.set_index('Disease'))

    # 🔍 Top Predictions
    st.subheader("🔍 Top Predictions")

    top_indices = np.argsort(prediction[0])[::-1][:3]

    for i in top_indices:
        st.write(f"{class_names[i]}: {prediction[0][i]*100:.2f}%")

    # 📊 Detailed Confidence
    st.subheader("📊 Detailed Confidence")

    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")
        st.progress(float(prob))

    # 📊 Confusion Matrix
    st.subheader("📊 Confusion Matrix (Demo)")

    y_true = np.random.randint(0, len(class_names), size=50)
    y_pred = np.random.randint(0, len(class_names), size=50)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.write("---")
st.caption("CNN Model | Streamlit App")