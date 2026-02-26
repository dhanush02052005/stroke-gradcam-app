import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import gdown
import os

# =========================
# Config
# =========================
IMG_SIZE = 224
CLASS_NAMES = ["No Stroke", "Ischemic", "Hemorrhagic"]
LAST_CONV_LAYER = "conv5_block16_concat"
MODEL_PATH = "stroke_densenet_model.keras"

st.title("🧠 Brain Stroke Detection with Grad-CAM")

# =========================
# Download model from Drive (ONLY if missing)
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model... please wait"):
            gdown.download(
                "https://drive.google.com/uc?id=YOUR_FILE_ID_HERE",
                MODEL_PATH,
                quiet=False
            )

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# Grad-CAM
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, class_index):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()

# =========================
# Upload UI
# =========================
uploaded_file = st.file_uploader("Upload CT Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_img = np.array(image)

    # preprocess
    img = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.densenet.preprocess_input(img)
    img_array = np.expand_dims(img, axis=0)

    # prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0])

    st.subheader(f"Prediction: {CLASS_NAMES[pred_class]}")
    st.write(f"Confidence: {confidence:.2%}")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(
        img_array, model, LAST_CONV_LAYER, pred_class
    )

    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32),
        (orig_img.shape[1], orig_img.shape[0])
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.image(orig_img, caption="Original Image", use_container_width=True)

    with col2:
        st.image(overlay, caption="Grad-CAM", use_container_width=True)
