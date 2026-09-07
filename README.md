# AegisCard - Credit Card Fraud Detection System

A production-ready machine learning framework designed to detect fraudulent credit card transactions using state-of-the-art classification techniques, data-leakage-free preprocessing, feature engineering, and optimal probability threshold tuning.

---

## Dataset Overview
The dataset used in this project is the Kaggle **Credit Card Fraud Detection Dataset** (`creditcard.csv`).
* **Total Transactions:** 284,807
* **Features:** 30 numerical variables (`V1` to `V28` PCA features, `Time`, `Amount`, and target `Class`)
* **Imbalance:** Severe class imbalance (Fraud accounts for **0.172%** of total transactions)

---

## Key Improvements & Architecture

### 1. Leakage-Free Preprocessing
* Scalers (`StandardScaler`) are fitted **strictly on `X_train`** after stratified splitting, preventing validation evaluation data leakage.

### 2. Domain Feature Engineering
* **Cyclical Hour of Day**: Extracted `Hour = (Time / 3600) % 24` to capture daily transaction activity patterns.
* **Log Amount Transformation**: Applied `log_amount = np.log1p(Amount)` to reduce extreme right skewness.

### 3. Evaluation Metrics & Threshold Tuning
* Prioritized **AUPRC (Area Under Precision-Recall Curve)** and **F1-Score** over misleading accuracy/ROC-AUC metrics.
* Automated precision-recall decision threshold search to maximize minority class detection.

### 4. Modular Python Architecture
* Clean separation of concerns into a reusable `src` package.

---

## Benchmark Results

| Model | ROC-AUC | AUPRC | Optimal Threshold | Max F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | `0.9520` | `0.8626` | `0.5300` | `0.8541` |
| **XGBoost** | **`0.9743`** | **`0.8673`** | `0.9541` | **`0.8571`** |

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
To run the automated pipeline (loads data, transforms features, fits models, tunes thresholds, and exports serialized models):
```bash
python -m src.train
```

### 3. Run Interactive Analysis Notebook
Open `fraud_detection.ipynb` in Jupyter Notebook or VS Code to visually explore confusion matrices and Precision-Recall curves.