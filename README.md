# 🏙️ CondoPrediction - Condominium Price Prediction Web App

A machine learning-powered web application that predicts the **estimated selling price of a condominium** based on location, nearby landmarks, and building features. Built using **Django**, **XGBoost**, and **JavaScript**, this project provides a fast, user-friendly interface for real estate price estimation.

## 🚀 Features

- 📍 Location-based inputs (District, Subdistrict, Road)
- 🏬 Distance to key landmarks (MRT/BTS, universities, airports, hospitals, malls, etc.)
- 🏢 Condo specifications: total units, age, and amenities (swimming pool, gym, security, etc.)
- 📊 Accurate predictions using a trained XGBoost regression model
- 🖥️ Fully deployed frontend with dynamic form controls and result display

---

## 🧠 Machine Learning Model

- Model: **XGBoost Regressor**
- Training data: Preprocessed dataset with location hierarchies and binary/categorical features
- Preprocessing: Label Encoding for categorical variables
- Model files:
  - `xgb_model.json`: Trained model
  - `label_encoders.pkl`: Encoded mappings for features

---

## 🛠️ Tech Stack

- **Backend**: Django, Python
- **Frontend**: HTML, CSS, JavaScript (AJAX for dynamic dropdowns)
- **ML**: XGBoost
- **Deployment**: Vercel / Localhost

---



