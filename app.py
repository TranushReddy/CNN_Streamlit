import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load the trained model
model = joblib.load(
    "C:\\Users\\tranu\\Desktop\\Tekwork-2\\CNN_Streamlit\\model\\cifar10_cnn_model.pkl"
)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# one hot encode the labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define class names for CIFAR-10
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Streamlit app
st.title("Image Classification with CNN")

# Select an image from the test set
image_index = st.slider(
    "Select an image index from the test set", 0, len(x_test) - 1, 0
)
selected_image = x_test[image_index]

# Display the selected image
st.image(
    selected_image,
    caption=f"Selected Image Index: {image_index}",
    use_column_width=True,
)

# Preprocess the image for prediction
input_image = np.expand_dims(selected_image, axis=0)

# Make a prediction using the loaded model
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)

# Display the predicted class
st.write(f"Predicted Class: {class_names[predicted_class]}")

# Display the true class
true_class = np.argmax(y_test[image_index])
st.write(f"True Class: {class_names[true_class]}")

# Display the accuracy of the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
st.write(f"Test Accuracy: {test_acc:.4f}")
st.subheader("Predictions on Sample Test Images")

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

images = x_test[:10]
true_labels = np.argmax(y_test[:10], axis=1)

preds = model.predict(images)
pred_labels = np.argmax(preds, axis=1)

fig, ax = plt.subplots(2, 5, figsize=(15, 5))
for i in range(10):
    ax[i // 5, i % 5].imshow(images[i])
    ax[i // 5, i % 5].set_title(
        f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[true_labels[i]]}"
    )
    ax[i // 5, i % 5].axis("off")
fig.tight_layout()
st.pyplot(fig)
