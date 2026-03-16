# 🌸 Iris Smart Predictor: A Production-Grade Neural Network Pipeline

[![Streamlit App](https://static.streamlit.io)](https://2dtlmfaqck5gkbkob6nqjg.streamlit.app)

An end-to-end Machine Learning project that classifies Iris flower species using a custom-built, scalable Deep Neural Network. This project demonstrates a full AI lifecycle, from raw data preprocessing to cloud deployment.

## 🚀 Live Demo
Check out the live application here: [Iris Smart Predictor](https://2dtlmfaqck5gkbkob6nqjg.streamlit.app)

## 🧠 Project Architecture & Features
Unlike basic ML tutorials, this project follows **Industrial AI Engineering** standards:

- **Scalable Architecture**: Built using a custom `IrisSmartNN` class in PyTorch, supporting dynamic hidden layers and BatchNorm/Dropout for better generalization.
- **Robust Preprocessing**: Implements `StandardScaler` to ensure feature consistency between training and real-world inference.
- **Inference Pipeline**: A modular approach separating the model definition (`model.py`) from the application logic (`app.py`).
- **Cloud Deployment**: Fully containerized-style deployment on Streamlit Cloud with an optimized dependency footprint.

## 🛠️ Step-by-Step Development
1. **Data Engineering**: Performed EDA and split data into **Train, Validation, and Test** sets to prevent data leakage.
2. **Model Design**: Developed a multi-layer perceptron (MLP) with Batch Normalization to accelerate convergence.
3. **Training**: Used **Adam Optimizer** and **CrossEntropyLoss**, achieving a **99.92% confidence level** on test samples.
4. **Serialization**: Saved model weights (`.pth`) and pre-trained scalers (`.joblib`) for efficient re-loading.
5. **Deployment**: Built a Streamlit UI providing real-time predictions with confidence scores.

## 📁 Repository Structure
- `app.py`: Main Streamlit application.
- `model.py`: PyTorch Neural Network architecture.
- `iris_smart_model.pth`: Trained model weights.
- `scaler.joblib`: Pre-trained StandardScaler object.
- `requirements.txt`: Project dependencies.

## 📓 Notebook
Explore the full training logic and experiments here: [Google Colab](https://colab.research.google.com/drive/1NzwMTwiI47u-8PNV2Y3TLBoFpycnNG-k?usp=sharing)
