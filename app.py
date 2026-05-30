import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="CNN CIFAR-10 Classifier", layout="wide")

st.title("Image Classification")
st.write("Predict image classes using a pre-trained CNN model")

# ---------------- LOAD MODEL ----------------


@st.cache_resource
def load_cnn_model():
    return load_model("model/cifar10_cnn_model.keras")


# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():
    return joblib.load("data/cifar10_test.pkl")


model = load_cnn_model()
x_test, y_test = load_data()

# ---------------- PREPROCESS ----------------

x_test = x_test.astype("float32") / 255.0

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

# ---------------- SIDEBAR ----------------

st.sidebar.header("Image Selection")

image_index = st.sidebar.slider(
    "Choose Test Image",
    min_value=0,
    max_value=len(x_test) - 1,
    value=0,
)

# ---------------- DISPLAY IMAGE ----------------

selected_image = x_test[image_index]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected Image")
    st.image(selected_image, width=300)

# ---------------- PREDICTION ----------------

input_image = np.expand_dims(selected_image, axis=0)

prediction = model.predict(input_image, verbose=0)

predicted_class = np.argmax(prediction)

true_class = int(y_test[image_index])

with col2:
    st.subheader("Prediction Result")

    st.success(f"Predicted Class: {class_names[predicted_class]}")

    st.info(f"True Class: {class_names[true_class]}")

# ---------------- SAMPLE PREDICTIONS ----------------

st.markdown("---")
st.subheader("Predictions on Sample Images")

sample_images = x_test[:10]
sample_labels = y_test[:10].flatten()

preds = model.predict(sample_images, verbose=0)
pred_labels = np.argmax(preds, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(10):

    ax = axes[i // 5, i % 5]

    ax.imshow(sample_images[i])

    ax.set_title(
        f"P: {class_names[pred_labels[i]]}\nT: {class_names[sample_labels[i]]}",
        fontsize=8,
    )

    ax.axis("off")

plt.tight_layout()

st.pyplot(fig)

# ---------------- ACCURACY ----------------

st.markdown("---")
st.subheader("Model Performance")

try:

    if len(y_test.shape) == 1:
        y_test_cat = np.eye(10)[y_test]
    else:
        y_test_cat = y_test

    loss, accuracy = model.evaluate(
        x_test,
        y_test_cat,
        verbose=0,
    )

    st.metric(
        "Test Accuracy",
        f"{accuracy:.4f}",
    )

except Exception as e:

    st.warning(f"Could not compute accuracy.\n\n{e}")
