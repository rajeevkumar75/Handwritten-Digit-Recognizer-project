import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image

# --- Load Model & Data ---
MODEL_PATH = "models/mnist_cnn.h5"
model = load_model(MODEL_PATH)

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

# --- Page Config ---
st.set_page_config(
    page_title="🖊️ Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.header("⚙️ Options")
mode = st.sidebar.radio("Choose Input Mode:", ["📤 Upload Image", "🎲 Use Sample Example"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Upload your own digit OR explore preloaded MNIST samples.")

# --- Main Title ---
st.title("🖊️ Handwritten Digit Recognizer")
st.caption("A **Deep Learning CNN Model** trained on the MNIST dataset")
st.markdown("---")

# --- Prediction Function ---
def predict_digit(img_array):
    preds = model.predict(np.expand_dims(img_array, axis=0))
    predicted_label = np.argmax(preds)
    return predicted_label, preds[0]

# --- Upload Mode ---
if mode == "📤 Upload Image":
    st.subheader("Upload your handwritten digit (28x28 grayscale preferred)")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, -1)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="🖼️ Uploaded Digit", width=150)

        with col2:
            pred, probs = predict_digit(img_array)
            st.success(f"### ✅ Predicted Digit: **{pred}**")
            st.markdown("#### 📊 Prediction Probabilities")
            st.bar_chart(probs)

# --- Sample Example Mode ---
elif mode == "🎲 Use Sample Example":
    st.subheader("Try a sample digit from MNIST test set")
    sample_idx = st.slider("Choose sample index", 0, 9999, 0)
    sample_img, true_label = x_test[sample_idx], y_test[sample_idx]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(sample_img.squeeze(), caption=f"🖼️ True Label: {true_label}", width=150)

    with col2:
        pred, probs = predict_digit(sample_img)
        if pred == true_label:
            st.success(f"### ✅ Predicted Digit: **{pred}** (Correct ✅)")
        else:
            st.error(f"### ❌ Predicted Digit: **{pred}** (True: {true_label})")

        st.markdown("#### 📊 Prediction Probabilities")
        st.bar_chart(probs)

# --- Footer ---
st.markdown("---")
st.caption("🚀 Built with TensorFlow, Keras, and Streamlit | MNIST Dataset")
