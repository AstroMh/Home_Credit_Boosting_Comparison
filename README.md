# Home Credit Default Risk — Gradient Boosting Comparison  
A full end-to-end machine learning project comparing **XGBoost**, **LightGBM**, and **CatBoost** on the real-world **Home Credit Default Risk** dataset from Kaggle.

This repository demonstrates:
- Advanced data preprocessing and feature engineering  
- Handling mixed numerical + categorical data  
- Training three state-of-the-art boosting models  
- Model evaluation and comparison (AUC, accuracy, F1, logloss, etc.)  
- Feature importance analysis  
- SHAP explainability for all three models  
- Runtime and resource comparison  
- Professional project structure suitable for portfolios

---

## 1. Project Overview

The goal of this project is to build a clean, reproducible pipeline that:
1. Loads and processes the **Home Credit Default Risk** dataset  
2. Trains three gradient boosting models:  
   - **XGBoost**  
   - **LightGBM**  
   - **CatBoost**
3. Evaluates each model using a consistent validation methodology  
4. Compares performance, speed, and interpretability  
5. Provides visual outputs and SHAP-based explanations  

This is a complete applied ML project using a large real dataset.

---

## 2. Repository Structure

home-credit-boosting-comparison/
│
├── data/
│ ├── raw/ # original Kaggle files
│ └── processed/ # cleaned and feature-engineered datasets
│
├── notebooks/ # optional EDA or experimentation
│ └── 01_eda.ipynb
│
├── src/
│ ├── config.py # configs, paths, settings
│ ├── data_loading.py # raw data ingestion
│ ├── preprocessing.py # cleaning, merging, feature engineering
│ ├── train_xgboost.py
│ ├── train_catboost.py
│ ├── train_lightgbm.py
│ ├── evaluate.py # evaluation functions
│ ├── compare_models.py # final comparison script
│ └── utils.py # helper utilities
│
├── models/ # saved model artifacts
│ ├── xgboost_model.json
│ ├── catboost_model.cbm
│ └── lightgbm_model.txt
│
├── results/
│ ├── eda/
│ ├── metrics/
│ ├── feature_importance/
│ ├── shap/
│ └── runtime/
│
├── reports/
│ └── model_comparison.md
│
├── requirements.txt
├── .gitignore
└── README.md


---

## 3. Dataset

This project uses the **Home Credit Default Risk** dataset from Kaggle:  
https://www.kaggle.com/competitions/home-credit-default-risk

**Files used:**
- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `POS_CASH_balance.csv`
- `credit_card_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`

> Place all downloaded files into `data/raw/`  
> (Instructions below in "How to Download the Dataset")

---

## 4. Installation

Clone the repository:

```bash
git clone https://github.com/Astr0Mh/home-credit-boosting-comparison.git
cd home-credit-boosting-comparison
```
Install dependencies:
pip install -r requirements.txt

## 5. Running the Pipeline

Preprocess the data:
```bash
python src/preprocessing.py
```

Train models:
```bash
python src/train_xgboost.py
python src/train_catboost.py
python src/train_lightgbm.py
```

Compare results:
```bash
python src/compare_models.py
```

## 6. Results

All results are saved under results/, including:
- metrics (AUC, accuracy, F1, RMSE)
- feature importance
- SHAP plots
- runtime logs
- EDA figures
- The final comparison table is saved in:
- results/metrics/model_comparison.csv

