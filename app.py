import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "healthcare_cnn_model.keras"
)

class_names = [
    "Normal",
    "Pneumonia"
]

st.title(
    "Chest X-Ray Classification"
)

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image"
    )

    image = image.resize((224,224))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(image)

    predicted_class = (
        prediction > 0.5
    ).astype(int)[0][0]

    st.success(
        f"Prediction: {class_names[predicted_class]}"
    )