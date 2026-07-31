# 🚦 AI Smart Queue Predictor & Intelligent Service Optimization System

---

## 📌 Project Overview

AI Smart Queue Predictor is a Machine Learning project that predicts customer waiting time using historical queue data.

The goal of this project is to help organizations reduce waiting time and improve service efficiency through data-driven decision making.

---

## 🎯 Problem Statement

Long customer queues reduce service quality and customer satisfaction.

Managers often struggle to estimate waiting times because they rely on manual judgment instead of historical data.

This project predicts waiting time using Machine Learning so organizations can better manage queues and allocate staff efficiently.

---

## 🌍 Real World Applications

- 🏥 Hospitals
- 🏦 Banks
- 🛂 Government Offices
- 🛒 Retail Stores
- ✈️ Airports
- 🎟️ Service Centers

---

## 📂 Dataset Features

| Feature | Description |
|----------|-------------|
| Service_Type | Type of service |
| Department | Department name |
| Current_Queue | Number of customers waiting |
| Available_Staff | Staff currently available |
| Emergency_Cases | Emergency cases |
| Previous_Average_Wait | Historical waiting time |
| Day | Day of week |
| Hour | Hour of day |
| Waiting_Time | Target Variable |

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
- Joblib

---

## 🤖 Machine Learning Models

This project compares three regression models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The best-performing model is saved and used in the Streamlit application.

---

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Model Comparison
8. Model Saving
9. Streamlit Deployment

---

## 📁 Project Structure

```
AI-Smart-Queue-Predictor/

│── app.py
│── queue_dataset.csv
│── queue_predictor.pkl
│── AI_Smart_Queue_Predictor.ipynb
│── requirements.txt
│── README.md
│── LICENSE
│── .gitignore

│── assets/
│     └── banner.png

│── screenshots/
│     ├── home.png
│     ├── prediction.png
│     └── result.png
```

---

## ▶️ Installation

Clone Repository

```bash
https://ai-smart-queue-predictor.streamlit.app/
```

Open Project

```bash
cd AI-Smart-Queue-Predictor
```

Install Libraries

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```
---

## 📈 Future Improvements

- Real-time Queue Monitoring
- Staff Recommendation
- Live Dashboard
- Cloud Deployment
- Mobile Support

---

## 👨‍💻 Author

**Muhammad Farhan**

BS Information Technology

GitHub:
https://github.com/farhan-ml

LinkedIn:
www.linkedin.com/in/muhammad-farhan-421715342

---

## 📜 License

This project is licensed under the MIT License.
