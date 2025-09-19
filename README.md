# 📊 Customer Churn Prediction  

This project is an **end-to-end Machine Learning pipeline** to predict whether a telecom customer will **churn (leave the service)** or not.  

The solution covers everything from **EDA → Model Building → Hyperparameter Tuning → Threshold Optimization → Deployment with Streamlit**, and is structured following **data science best practices**.  

---

## 🚀 Project Overview  

Customer churn is a critical business problem for telecom companies. Retaining existing customers is often more cost-effective than acquiring new ones.  

This project answers:  
- Which customers are likely to churn?  
- What are the most important factors driving churn?  
- How can we balance **recall (catching churners)** vs **precision (avoiding false alarms)**?  

We used machine learning models to predict churn and deployed the best-performing model using **Streamlit** for easy interaction.  

---

## 🛠️ Tech Stack  

- **Language**: Python 3.9+  
- **Libraries**:  
  - `pandas`, `numpy` → Data wrangling  
  - `matplotlib`, `seaborn` → EDA & visualization  
  - `scikit-learn` → Model building, preprocessing, metrics  
  - `imblearn` → SMOTE for imbalance handling  
  - `xgboost` → Gradient boosting  
  - `joblib` → Model serialization  
  - `streamlit` → Deployment  

---

## 📂 Project Workflow  

### 1. Exploratory Data Analysis (EDA)  
- Checked missing values, duplicates, and data types.  
- Converted categorical variables (e.g., `Yes/No`, `Male/Female`) into binary features.  
- Univariate & bivariate analysis to study churn patterns.  
- Identified imbalance: ~26% churn rate.  

### 2. Data Preprocessing  
- Binary mapping (`Yes/No`, `Male/Female` → 0/1).  
- One-hot encoding for multi-class categorical variables.  
- Standard scaling for numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`).  
- SMOTE applied to balance churn vs non-churn classes.  

### 3. Model Building  
- **Baseline model**: Logistic Regression (with class weights).  
- **Tree-based models**: Decision Tree, Random Forest, XGBoost.  
- Compared models using: Accuracy, Precision, Recall, F1-score, ROC-AUC.  

### 4. Model Tuning & Optimization  
- **RandomizedSearchCV / GridSearchCV** for hyperparameter tuning (Random Forest & XGBoost).  
- **Threshold tuning** using Precision-Recall tradeoff to find optimal decision threshold (instead of default 0.5).  
- Best-performing model: **Random Forest with threshold = 0.45**.  

### 5. Deployment  
- Saved model, scaler, and threshold using `joblib`.  
- Built an interactive **Streamlit web app** where users can input customer details and get churn predictions with probability scores.  

---

## 📈 Results  

| Model              | Accuracy | Recall (Churn) | Precision (Churn) | F1 (Churn) | ROC-AUC |
|--------------------|----------|----------------|--------------------|------------|---------|
| Logistic Regression| ~0.75    | 0.79           | 0.51               | 0.62       | 0.83    |
| Decision Tree      | ~0.71    | 0.80           | 0.47               | 0.59       | 0.82    |
| Random Forest (Tuned + Threshold=0.45) | ~0.76 | 0.75 | 0.54 | 0.62 | 0.83 |
| XGBoost (Tuned + Threshold=0.72) | ~0.75 | 0.76 | 0.52 | 0.62 | 0.82 |

✅ **Final choice**: Random Forest (tuned + threshold optimized) → best tradeoff between recall & precision.  

---

## 🎯 Key Learnings  

- Threshold tuning significantly improved the **recall vs precision tradeoff**.  
- Class imbalance handling (SMOTE) was critical to avoid bias toward the majority class.  
- Random Forest provided interpretable feature importance, helping understand churn drivers (e.g., contract type, tenure, charges).  
- Deployment with Streamlit made the project more **resume-ready and business-friendly**.  

---

## 📌 Next Steps  

- 🏗️ Explore advanced ensemble methods (Stacking, LightGBM, CatBoost).  
- ☁️ Deploy on cloud platforms (Heroku, AWS, GCP) for scalability.  
- 📊 Implement model monitoring pipelines to detect **data drift** and **model drift**.  
- 🔍 Perform customer segmentation to design targeted retention strategies.  
- 📈 Add SHAP or LIME for better model explainability.  

---

## 🧑‍💻 Author  

👤 **Muhammad Basharat Asghar**  
- Data Scientist | Machine Learning Practitioner | End-to-End Projects  
- [LinkedIn](https://www.linkedin.com/in/basharat-asghar/)  
