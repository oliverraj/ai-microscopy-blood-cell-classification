# AI-Assisted Microscopy Blood Cell Classification

> End-to-end deep learning pipeline for automated blood cell classification using **PyTorch**, **OpenCV**, **Grad-CAM**, and **Streamlit**.

---

## Project Overview

This project demonstrates an end-to-end deep learning workflow for automated blood cell classification from peripheral blood smear microscopy images.

The application combines classical computer vision, transfer learning, explainable AI, and interactive deployment into a single healthcare AI pipeline.

Users can upload microscopy images through an interactive Streamlit application, obtain model predictions, visualize prediction confidence, and interpret model decisions using Grad-CAM heatmaps.

> **Disclaimer:** This project is intended for educational and portfolio demonstration purposes only. It is **not** intended for clinical diagnosis or patient care.

---

## Features

- Dataset auditing and exploratory data analysis (EDA)
- OpenCV image preprocessing pipeline
- PyTorch Dataset and DataLoader pipeline
- Transfer Learning using pretrained ResNet18
- Deep learning model training
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - Macro F1 Score
  - Confusion Matrix
- Grad-CAM explainability
- Interactive Streamlit web application

---

## Dataset

**Dataset**

Blood Cells Image Dataset

- 17,092 microscopy images
- RGB images
- Approximate image size: 360 × 363 pixels

### Cell Classes

- Basophil
- Eosinophil
- Erythroblast
- Immature Granulocyte (IG)
- Lymphocyte
- Monocyte
- Neutrophil
- Platelet

---

## Project Workflow

```
Microscopy Images
        │
        ▼
Dataset Audit & EDA
        │
        ▼
OpenCV Preprocessing
        │
        ▼
PyTorch Dataset Pipeline
        │
        ▼
Transfer Learning (ResNet18)
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Inference Pipeline
        │
        ▼
Grad-CAM Explainability
        │
        ▼
Interactive Streamlit Application
```

---

## Project Structure

```
ai-microscopy-blood-cell-classification/

├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│
├── models/
│
├── reports/
│   ├── figures/
│   ├── evaluation_metrics.txt
│   ├── training_history.csv
│   └── classification_report.csv
│
├── notebooks/
│
└── src/
    ├── classical_cv.py
    ├── dataset.py
    ├── eda.py
    ├── evaluate.py
    ├── gradcam.py
    ├── inference.py
    ├── model.py
    └── train_resnet.py
```

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Model | ResNet18 |
| Framework | PyTorch |
| Transfer Learning | ImageNet Pretrained |
| Input Size | 224 × 224 |
| Optimizer | Adam |
| Loss Function | CrossEntropy Loss |
| Output Classes | 8 |

---

## Model Performance

Validation Results

| Metric | Score |
|---------|-------|
| Accuracy | **80.93%** |
| Precision | **83.75%** |
| Recall | **76.68%** |
| Macro F1 Score | **78.37%** |

The model demonstrates strong classification performance across eight blood cell categories using transfer learning.

---

## Explainable AI (Grad-CAM)

This project incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations of model predictions.

Grad-CAM highlights image regions that contributed most strongly to the predicted class, allowing users to better understand the model's decision-making process.

For neutrophil predictions, the model primarily focuses on the segmented nucleus and surrounding cytoplasmic morphology rather than background erythrocytes, demonstrating biologically meaningful attention.

---

## Interactive Streamlit Application

The Streamlit application allows users to:

- Upload microscopy images
- Predict blood cell type
- View prediction confidence
- Compare class probabilities
- Visualize Grad-CAM heatmaps
- Review model explanations interactively

---

## Technologies Used

### Programming

- Python

### Deep Learning

- PyTorch
- TorchVision

### Computer Vision

- OpenCV
- Pillow

### Data Analysis

- NumPy
- Pandas
- Scikit-learn

### Visualization

- Matplotlib

### Deployment

- Streamlit

---

## Example Outputs

The project generates:

- Dataset audit report
- Class distribution plots
- Image size analysis
- OpenCV preprocessing pipeline
- Training loss curves
- Training accuracy curves
- Confusion matrix
- Classification report
- Grad-CAM visualizations
- Interactive prediction dashboard

---

## Future Improvements

Potential future enhancements include:

- Vision Transformer (ViT) comparison
- EfficientNet implementation
- Model ensemble methods
- Confidence calibration
- Docker deployment
- Cloud deployment
- Multi-disease blood smear classification
- Integration with digital pathology workflows

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-microscopy-blood-cell-classification.git

cd ai-microscopy-blood-cell-classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Clinical Relevance

Automated blood cell classification has applications in hematology, pathology, and laboratory medicine.

AI-assisted microscopy can support:

- Faster screening workflows
- Improved consistency
- Reduced manual review burden
- Educational decision-support systems

This project demonstrates how deep learning and explainable AI techniques can be combined to create transparent image classification workflows for biomedical imaging.

---

## Author

**Rajeev Prasad**

Master of Applied Data Science (MADS)

University of Michigan

Healthcare AI • Biomedical Data Science • Machine Learning • Medical Imaging

---

## License

This project is released under the MIT License.

---

## Acknowledgments

- University of Michigan – Master of Applied Data Science
- PyTorch
- OpenCV
- Streamlit
- Kaggle Blood Cells Image Dataset