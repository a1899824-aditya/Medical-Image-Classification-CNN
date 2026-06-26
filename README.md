# Medical Image Classification CNN

Deep learning project for automated pneumonia detection from chest X-ray images using custom Convolutional Neural Networks (CNNs) and transfer learning with ResNet50.

---

## Project Overview

This project investigates multiple deep learning approaches for binary classification of chest X-ray images into:

- **NORMAL**
- **PNEUMONIA**

The project follows a complete machine learning workflow including:

- Dataset exploration
- Image preprocessing
- Data augmentation
- Development of multiple custom CNN architectures
- Handling class imbalance
- Model evaluation using multiple performance metrics
- Transfer learning with ResNet50
- Fine-tuning experiments
- Streamlit deployment

---

## Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

Source:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset is **not included** in this repository because it is approximately **1.16 GB**, which exceeds GitHub's recommended file size limits.

Expected folder structure:

```text
data/
└── chest_xray/
    ├── train/
    ├── test/
    └── val/
```

---

## Models Evaluated

| Model | Accuracy | AUC | Description |
|:------|---------:|----:|:------------|
| Baseline CNN | 80.00% | 0.911 | Initial CNN model |
| CNN V1 | 76.88% | - | Reduced complexity architecture |
| CNN V2 | 80.94% | 0.936 | Improved CNN with Batch Normalization |
| CNN V3 | 85.78% | 0.941 | Best custom CNN |
| Class Weighted CNN | 80.94% | - | CNN trained using class weights |
| Frozen ResNet50 | **86.86%** | **0.952** | Final selected model |
| Fine-tuned ResNet50 (Last 5 Layers) | 77.56% | 0.957 | Fine-tuning experiment |
| Fine-tuned ResNet50 (Last 10 Layers) | 75.64% | 0.956 | Fine-tuning experiment |

---

## Best Model

The **Frozen ResNet50** model achieved the best balance between accuracy, robustness and generalisation.

### Final Performance

- **Accuracy:** 86.86%
- **AUC:** 0.952
- **Selected Model:** Frozen ResNet50

Although fine-tuning increased AUC slightly, it reduced overall test accuracy. Therefore, the frozen ResNet50 model was selected as the final model.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Pillow
- Jupyter Notebook
- Streamlit
- ResNet50 Transfer Learning

---

## Installation

Clone the repository:

```bash
git clone https://github.com/a1899824-aditya/Medical-Image-Classification-CNN.git
```

Move into the project directory:

```bash
cd Medical-Image-Classification-CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
Medical_Image_Classification_CNN.ipynb
```

---

## Running the Streamlit App

```bash
streamlit run pneumonia_app.py
```

The application allows users to upload a chest X-ray image and receive a prediction indicating whether the image is classified as **NORMAL** or **PNEUMONIA**.

---

## Repository Structure

```text
Medical-Image-Classification-CNN/
│
├── data/
│   └── README.md
│
├── Medical_Image_Classification_CNN.ipynb
├── pneumonia_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Future Improvements

- Deploy the application online
- Implement Grad-CAM visual explanations
- Evaluate EfficientNet and DenseNet architectures
- Perform k-fold cross-validation
- Train on larger medical imaging datasets
- Improve model explainability

---

## Disclaimer

This project was developed for educational and portfolio purposes only.

It is **not** intended for clinical diagnosis or medical decision-making.

---

## Author

**Aditya Venugopalan**

Master of Data Science

University of Adelaide
