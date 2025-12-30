# Customer Churn Prediction using Deep Learning (ANN)

## Project Overview

This project focuses on **predicting customer churn** using a **Deep Learning Artificial Neural Network (ANN)** built with **Keras (Sequential API)** and trained using **TensorFlow**.

The complete workflow includes:
- Data preprocessing
- Feature encoding & scaling
- ANN model training
- Saving trained components
- Reusing saved models and preprocessors for prediction


---

## Problem Statement

Customer churn occurs when users stop using a product or service.  
The objective of this project is to **predict whether a customer will churn or not** based on historical customer data.

---

## Tech Stack Used
```
- Python
- TensorFlow
- Keras (Sequential ANN model)
- Scikit-learn
- Pandas, NumPy
- Jupyter Notebook
```
---
## 📂 Project Structure
```
churnPrediction/
│
├── notebooks/
│ └── datasets/
│ │   ├── Churn_Modelling.csv # Original dataset
│ │   └── datafile.zip # Dataset archive
│ ├── X_data.csv # Processed input features
│ ├── y_data.csv # Target labels
│ ├── data.ipynb # Data preprocessing & EDA
│ ├── model.ipynb # ANN model creation & training
│ ├── prediction.ipynb # Inference using saved models
│ ├── gender_label.pkl # Label encoder for Gender
│ ├── geography_label.pkl # Label encoder for Geography
│ └── scaler.pkl # Feature scaler
│── requirements.txt
│── .gitignore
└── README.md
```
---
## Project Workflow

### 1️. Data Preprocessing (`data.ipynb`)
- Loaded customer churn dataset
- Handled categorical features
- Applied **Label Encoding**
- Feature scaling using **StandardScaler**
- Saved encoders and scaler as `.pkl` files

---

### 2️. Model Training (`model.ipynb`)
- Built an **ANN using Keras Sequential API**
- Used:
  - Input layer
  - Hidden dense layers
  - Output layer with sigmoid activation
- Trained model using **TensorFlow backend**
- Evaluated model performance
- Saved trained model components for reuse

---

### 3️. Prediction (`prediction.ipynb`)
- Loaded saved:
  - Label encoders
  - Scaler
  - Trained ANN model
- Performed predictions on new/unseen data
- Ensured consistency with training pipeline

---

## Model Details

- **Model Type:** Artificial Neural Network (ANN)
- **Framework:** Keras + TensorFlow
- **Activation Functions:** ReLU, Sigmoid
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam

---

##  How to Run the Project

### Step 1: Clone the repository
```bash
git clone https://github.com/itshowrohitworks/churnPrediction.git
cd churnPrediction
```
---
### Step 2: Install dependencies
```
pip install -r requirements.txt
```
### Step 3: Run notebooks in order
```
data.ipynb

model.ipynb

prediction.ipynb
```
