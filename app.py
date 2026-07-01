import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import predict_image
from src.gradcam import generate_gradcam


st.set_page_config(
    page_title="AI Microscopy Blood Cell Classifier",
    layout="wide",
)

st.title("AI-Assisted Microscopy Blood Cell Classification")

st.markdown("""
This application demonstrates an AI-assisted microscopy image analysis workflow for blood cell classification.

The model uses a PyTorch ResNet18 transfer learning pipeline trained on microscopy images.
It is designed as a portfolio demonstration for medical imaging, cellular discrimination, and diagnostic AI workflows.

**Disclaimer:** This tool is for educational and portfolio demonstration only. It is not intended for clinical diagnosis.
""")

with st.sidebar:
    st.header("Project Summary")
    st.write("**Task:** Blood cell image classification")
    st.write("**Model:** ResNet18 transfer learning")
    st.write("**Framework:** PyTorch")
    st.write("**Classes:** 8")
    st.write("**Validation Accuracy:** 80.93%")
    st.write("**Explainability:** Grad-CAM")

uploaded_file = st.file_uploader(
    "Upload a blood cell microscopy image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = Path(temp_file.name)

    prediction = predict_image(temp_path)
    gradcam_result = generate_gradcam(temp_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        st.metric("Predicted Class", prediction["predicted_class"])
        st.metric("Confidence", f"{prediction['confidence']:.2%}")

        probabilities_df = pd.DataFrame(
            {
                "Class": list(prediction["probabilities"].keys()),
                "Probability": list(prediction["probabilities"].values()),
            }
        ).sort_values("Probability", ascending=False)

        st.subheader("Class Probabilities")
        st.dataframe(probabilities_df, use_container_width=True)
        st.bar_chart(probabilities_df.set_index("Class"))

    st.markdown("---")
    st.subheader("Grad-CAM Explainability")

    st.markdown("""
    Grad-CAM highlights image regions that contributed most strongly to the model prediction.
    In healthcare AI, visual explanations can help review whether the model is focusing on biologically meaningful regions.
    """)

    cam_col1, cam_col2, cam_col3 = st.columns(3)

    with cam_col1:
        st.image(gradcam_result["original"], caption="Original", use_container_width=True)

    with cam_col2:
        st.image(gradcam_result["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)

    with cam_col3:
        st.image(gradcam_result["overlay"], caption="Overlay", use_container_width=True)

st.markdown("---")

st.subheader("Project Workflow")

st.markdown("""
1. Dataset audit and class distribution analysis  
2. Classical computer vision preprocessing using OpenCV  
3. Transfer learning with ResNet18 in PyTorch  
4. Model evaluation using precision, recall, F1-score, and confusion matrix  
5. Grad-CAM explainability and interactive deployment  
""")

st.subheader("Clinical Relevance")

st.markdown("""
Microscopy-based cellular classification is important in diagnostic pathology and hematology.
AI-assisted image analysis can support consistency, reduce manual review burden, and provide scalable decision-support workflows.
""")