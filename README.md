# T-Mobile Retention Analytics Platform

This project is a **customer churn prediction platform** built with **Python, Scikit-learn, SHAP, and Streamlit**.  
It predicts the probability of customer churn for T-Mobile and provides **clear visual explanations and recommended business actions**.

Deployed Live:  
[T-Mobile Retention Analytics Platform](https://t-mobile-retention-analytics-platform.onrender.com)

---

## Features
- **Churn Probability Score**: Displays customer churn risk as a percentage with an intuitive bar chart gauge.
- **Key Drivers of Churn**: Visualises the top 5 features influencing the churn prediction (SHAP values).
- **Actionable Recommendations**: Provides frontline staff and executives with clear next steps to reduce churn.
- **Professional UI**: Clean T-Mobile–inspired dark theme with mobile-friendly layout.

---

## Why These Features?
Instead of overwhelming business users with all model inputs, the app highlights only:
1. **Churn Risk Score** → Easy-to-understand probability for executives and support teams.  
2. **Top 5 Drivers of Churn** → Focus on the most influential factors (e.g., contract type, tenure, charges).  
3. **Recommendations** → Direct next steps (e.g., “Offer a discount”, “Reinforce value before 2 years”).  

This balances **model interpretability** with **business usability**.

---

## Example Visuals
The app produces three core sections:

1. **Churn Risk Score**  
   - Shows a customer’s churn probability clearly.  
   - Example: `High Risk: This customer has a 39.9% chance of leaving.`  

2. **Why this prediction?**  
   - Top 5 churn drivers chart with **red (increasing churn)** and **blue (reducing churn)** bars.  
   - Helps staff see *why* a customer is at risk.  

3. **Recommended Actions**  
   - Actionable business strategies linked to the key churn drivers.  
   - Example:  
     - *Fiber Optic Internet* → “Check satisfaction — fiber customers often leave if expectations aren’t met.”  
     - *Tenure: 7–12 months* → “Proactively engage — many customers leave in their first year.”  

---

## Project Structure
```
BAN6800_FINAL_PROJECT/
│── app.py                      # Main Streamlit application
│── logistic_model_final.pkl     # Trained churn prediction model
│── scaler_final.pkl             # Pre-fitted scaler for preprocessing
│── shap_explainer_final.pkl     # SHAP explainer for model interpretability
│── requirements.txt             # Dependencies for deployment
│── Churn.ipynb                  # Jupyter notebook - data exploration
│── Churn_Model.ipynb            # Jupyter notebook - model training
│── telco_model_ready.csv        # Cleaned dataset for training
│── Churn Original Dataset.csv   # Raw dataset
```

---

## Installation (Local Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/BAN6800_FINAL_PROJECT.git
   cd BAN6800_FINAL_PROJECT
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate     # Mac/Linux
   venv\Scripts\activate        # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Deployment on Render

### 1. Push to GitHub
Ensure your repo contains:
- `app.py`  
- `requirements.txt`  
- Model artifacts (`.pkl` files)

### 2. Create Render Web Service
- Go to [Render](https://render.com) → **New Web Service**  
- Connect to your GitHub repo  

### 3. Configure Service
- **Environment**: Python 3.9+  
- **Build Command**:  
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:  
  ```bash
  streamlit run app.py --server.port 10000 --server.address 0.0.0.0
  ```

### 4. Deploy
Render builds the app automatically.  
Once live, your app is available here:  
[T-Mobile Retention Analytics Platform](https://t-mobile-retention-analytics-platform.onrender.com)

---

## Technologies Used
- **Python** (Scikit-learn, Pandas, Numpy)  
- **Streamlit** (UI & deployment)  
- **SHAP** (Explainable AI for churn drivers)  
- **Render** (Deployment platform)  

---

## Business Value
This platform enables:
- **Executives** → Understand customer churn trends with intuitive visuals.  
- **Frontline teams** → Receive concrete recommendations to engage at-risk customers.  
- **Data teams** → Ensure transparency with explainable AI (SHAP).  

By presenting churn scores alongside explanations and actions, the app bridges **machine learning insights** with **real business decision-making**.
