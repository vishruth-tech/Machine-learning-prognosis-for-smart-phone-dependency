# 📱 Machine Learning Prognosis for Smartphone Dependency

A machine learning web application that predicts a user's level of **smartphone addiction/dependency** based on their behavioral patterns. Built with Flask and powered by three ensemble ML models.

---

## 🎯 What It Does

Users answer a short questionnaire about their smartphone habits (e.g., checking phone without notifications, taking it to the bathroom, worrying about losing it). The app then runs the responses through three trained ML models and predicts the user's dependency level.

---

## 🧠 Models Used

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| Stacking Classifier | 100% | 94% |
| CatBoost Classifier | 99.84% | 96.12% |
| ExtraTrees Classifier | 99.84% | 100% |

> Models are pre-trained and stored as `.pkl` files in `CODE/model/`.

---

## 🛠 Tech Stack

- **Backend:** Python 3.12, Flask 3.0.3
- **ML:** scikit-learn, CatBoost, XGBoost
- **Data:** pandas, numpy
- **Frontend:** HTML, CSS (Bootstrap), JavaScript (jQuery)

---

## 📁 Project Structure

```
CODE/
├── app.py                  ← Flask entry point
├── requirements.txt        ← Python dependencies
├── model/
│   ├── et_model.pkl        ← ExtraTrees model
│   ├── stacking.pkl        ← Stacking classifier
│   ├── catboost.pkl        ← CatBoost model
│   └── smartphone_de.csv   ← Training dataset
├── templates/              ← HTML pages (index, prediction, result, etc.)
├── static/
│   ├── css/                ← Stylesheets
│   ├── js/                 ← Bootstrap + jQuery
│   └── images/             ← UI images and charts
└── test_data/
    └── test_data.csv       ← Sample test data
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/vishruth-tech/Machine-learning-prognosis-for-smart-phone-dependency.git
cd Machine-learning-prognosis-for-smart-phone-dependency
```

### 2. Install dependencies
```bash
cd CODE
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
Navigate to: **http://localhost:5000**

---

## 📊 Dataset

The model was trained on a survey dataset capturing smartphone usage behavior across students. Features include:

- Gender
- Taking pictures of class notes with phone
- Accessing books/study material via mobile
- Running for charger when battery dies
- Worry about losing phone
- Taking phone to bathroom
- Phone usage in social gatherings/parties
- Checking phone without any notification

---

## 🔮 Predictions

The app predicts whether a user is:
- **Addicted** to their smartphone
- **Not Addicted**

Based on ensemble voting across the three trained models.

---

## ⚠️ Notes

- Run the app from inside the `CODE/` directory so model paths resolve correctly.
- The app runs in debug mode by default (development use only). Set `FLASK_DEBUG=False` for production.
