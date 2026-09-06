# AegisCard - Credit Card Fraud Detection

A machine learning project designed to detect fraudulent credit card transactions using classification techniques.

## 📌 Dataset
The dataset used in this project is the Kaggle **Credit Card Fraud Detection Dataset** (`creditcard.csv`).
* **Total Transactions:** ~284,807
* **Features:** 30 numerical variables (`V1` to `V28` PCA features, `Time`, `Amount`, and target `Class`)
* **Imbalance:** Highly imbalanced (Fraud accounts for ~0.17% of total transactions)

## 📦 Dependencies & Prerequisites
This project requires Python 3.8+ and the following Python packages:

| Package | Purpose |
| :--- | :--- |
| **`pandas`** | Data manipulation and loading CSV files |
| **`numpy`** | Array handling and numerical calculations |
| **`scikit-learn`** | Machine learning models, preprocessing, and metrics |
| **`matplotlib`** | Plotting engine for data visualization |
| **`seaborn`** | Enhanced statistical graphics and heatmaps |
| **`jupyter`** | Interactive notebook environment for running `.ipynb` files |

## 📁 Project Structure
AegisCard/
├── creditcard.csv       # Dataset file
├── fraud_detection.ipynb # Analysis, training, & evaluation notebook
└── README.md            # Project documentation