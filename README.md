# Returns Intelligence AI

Predicting which e-commerce orders will be returned or canceled before they happen, using machine learning and customer review NLP.

## Live Interactive Dashboard

[View on Tableau Public](https://public.tableau.com/authoring/ReturnsintelligenceAI/Dashboard1#1)

## Project Overview

End-to-end data analytics and machine learning project on 99,441 real Brazilian e-commerce orders from 2016-2018. Analyzes return drivers, predicts bad outcomes from order-time features, and surfaces complaint themes from Portuguese customer reviews.

## Key Findings

- 99,441 orders analyzed across 9 linked datasets
- 14.69% bad outcome rate (cancellations + low review scores)
- R$ 2,644,652 revenue at risk (16.7% of total revenue)
- Worst categories by return rate: office_furniture (22.45%), audio (21.39%), bed_bath_table (16.23%)
- Worst states: AL (23.49%), SE (22.00%), MA (21.82%), RJ (20.56%)
- Delivery delay is the strongest driver: orders 15+ days late have a 75.96% bad outcome rate vs only 8.99% for on-time deliveries
- AI model: XGBoost classifier with 0.65 ROC-AUC predicting bad outcomes from order-time features

## Business Recommendations

1. Fix delivery logistics in northeastern Brazilian states (AL, SE, MA, RJ) where return rates exceed 20%
2. Quality-review the top 5 worst categories, especially office_furniture which loses R$ 87,000+ to returns
3. Deploy the ML risk score at checkout to flag high-risk orders for human review before fulfillment
4. Address the November-March seasonal return spike with extra QA staffing and tighter shipping SLAs

## Technical Approach

- Data cleaning and feature engineering in pandas across 9 source tables
- SQL business analysis with 7 queries on a SQLite database
- Machine learning with XGBoost trained on order-time features only
- Explainable AI using SHAP values
- NLP with TF-IDF and KMeans clustering on Portuguese reviews
- Visualization with matplotlib, plotly, and Tableau Public

## Tools

Python (pandas, numpy, scikit-learn, XGBoost, SHAP, NLTK, matplotlib, plotly), SQL (SQLite), Tableau Public, VS Code

## Dataset

[Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Author

Pranavi Gantla - [LinkedIn](https://www.linkedin.com/in/pranavi-g-942474361/)  
Email :- pranavig2002@gmail.com
