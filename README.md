# amazon_stock_data_XGBoost-and-KNN

### This repository focuses on predicting the stock price movement of Amazon.com, Inc. (AMZN) using two powerful machine learning algorithms: XGBoost and K-Nearest Neighbors (KNN). The goal is to determine whether the stock price will go higher or lower compared to the previous day.


#### 🌟 Key Features

1. Machine Learning Models

XGBoost: Optimized with hyperparameters (learning_rate=0.05, max_depth=5) for regression, achieving minimal RMSE.

KNN Classification: Predicts "Up" or "Down" trends using k=8 neighbors, validated by AUC scores.

Technical Indicators: RSI, ADX, and Parabolic SAR integrated to capture market trends.

2. Streamlit Web App

Interactive Dashboard: Visualize predictions, historical data, and model performance metrics.

User Inputs: Adjust parameters (e.g., forecast window, KNN neighbors) and test custom dates.

Live Updates: Dynamic charts display predictions vs. actual stock movements.

Model Insights: Explainability sections for XGBoost decision trees and KNN logic.

🛠️ Tech Stack
Backend: Python, XGBoost, Scikit-learn (KNN), Pandas, TA-Lib (technical indicators).

Frontend: Streamlit, Plotly (interactive graphs), Highcharter.

Deployment: Compatible with Streamlit Cloud, Docker, or local hosting.

📊 App Preview
![Demo GIF/Link] (Add a screenshot or link to your live app demo here)

🚀 How to Use
Clone the repo:

bash
git clone https://github.com/abhishek199677/amazon_stock_data_XGBoost-and-KNN

Install dependencies:

bash
pip install -r requirements.txt  
Launch the app:

bash
streamlit run app.py  
📌 Ideal For
Investors testing ML-driven stock strategies.

Developers learning to deploy financial models via Streamlit.

Educators teaching time-series forecasting with real-world data.

🔗 Live Demo
https://amazonxgknn.streamlit.app/

Contribution Welcome!

Improve UI/UX design for the Streamlit app.

Add new features like sentiment analysis or LSTM models.

Optimize data pipelines for faster predictions.

🔍 Check the app.py file for frontend logic and model_training.ipynb for ML workflows.














streamlit run app.py      