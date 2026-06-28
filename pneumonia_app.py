import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model
model = load_model("models/frozen_resnet50_winner.keras")

# Page title
st.title("🫁 Pneumonia Detection using ResNet50")

st.write(
    "Upload a chest X-ray image and the model will predict whether pneumonia is present."
)

st.warning(
    "This tool is for chest X-ray images only. It is not a medical diagnosis."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    st.image(
        uploaded_file,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    # Load image
    img = image.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    # Convert to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ResNet preprocessing
    img_array = preprocess_input(
        img_array
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    score = float(prediction[0][0])

    # Decision threshold
    THRESHOLD = 0.7

    st.subheader("Prediction Results")

    st.write(
        f"**Pneumonia Probability Score:** {score:.4f}"
    )

    if score > THRESHOLD:

        st.error(
            f"🚨 Prediction: PNEUMONIA ({score:.2%})"
        )

    else:

        st.success(
            f"✅ Prediction: NORMAL ({(1-score):.2%})"
        )

    st.info(
        "Predictions may be unreliable for non-X-ray images or poor-quality scans."
    )