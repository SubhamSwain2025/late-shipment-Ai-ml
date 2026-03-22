# 🚀 Shipment Delay Prediction API

An end-to-end Machine Learning system to predict shipment delays using **Random Forest**, **FastAPI**, and **MLflow**.

---

## 📌 Project Overview

This project predicts whether a shipment will be **late** or **very late** based on order details such as shipping mode, order value, location, and customer data.

It includes:

* Data preprocessing & feature engineering
* Machine learning model training
* Experiment tracking using MLflow
* FastAPI backend for real-time predictions

---

## 🔥 Features

* ✅ End-to-end ML pipeline (data → training → prediction)
* ✅ Dual model system:

  * Late shipment prediction
  * Very late shipment prediction
* ✅ FastAPI backend (REST API)
* ✅ MLflow experiment tracking
* ✅ Local + AWS-ready architecture

---

## 🧠 Tech Stack

* Python 🐍
* Scikit-learn
* FastAPI
* MLflow
* Pandas & NumPy

---

## 📂 Project Structure

```
late-shipment-ml/
│── api/                # FastAPI backend
│── data/               # Raw & processed data
│── models/             # Saved ML models
│── notebooks/          # Jupyter notebooks
│── src/                # Core ML logic
│── run_pipeline.py     # Training pipeline
│── retrain_pipeline.py
│── requirements.txt
```

---

## ▶️ Run Locally

### 1. Clone the repo

```
git clone https://github.com/SubhamSwain2025/late-shipment-Ai-ml.git
cd late-shipment-Ai-ml
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Train model

```
python run_pipeline.py
```

### 5. Run API

```
uvicorn api.main:app --reload
```

---

## 🌐 API Endpoints

| Endpoint              | Description                      |
| --------------------- | -------------------------------- |
| `/ping`               | Health check                     |
| `/predict_late/`      | Predict if shipment is late      |
| `/predict_very_late/` | Predict if shipment is very late |

---

## 🧪 Example Input

```json
{
  "order_item_quantity": 3,
  "order_item_total": 900,
  "product_price": 300,
  "year": 2026,
  "month": 7,
  "day": 15,
  "order_value": 900,
  "unique_items_per_order": 3,
  "order_item_discount_rate": 0.05,
  "units_per_order": 3,
  "order_profit_per_order": 80,
  "type": "Corporate",
  "customer_segment": "Corporate",
  "shipping_mode": "First Class",
  "category_id": 2,
  "customer_country": "India",
  "customer_state": "Karnataka",
  "department_id": 1,
  "order_city": "Bangalore",
  "order_country": "India",
  "order_region": "Southern Asia",
  "order_state": "Karnataka"
}
```

---

## 📊 Example Output

```json
{
  "late_prediction": 1
}
```

---

## 🎯 Use Case

* Logistics optimization
* Delivery risk prediction
* Supply chain analytics

---

## 📄 Resume Highlight

**Developed an end-to-end ML-based shipment delay prediction system using Python, Scikit-learn, FastAPI, and MLflow with real-time API deployment.**

---

## 👨‍💻 Author

**Subham Swain**
GitHub: https://github.com/SubhamSwain2025
