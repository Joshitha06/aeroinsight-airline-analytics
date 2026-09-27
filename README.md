# ✈️ AeroInsight — Airline Passenger Satisfaction Analytics

Independent data analytics project exploring what drives airline passenger satisfaction, using a public dataset. Built to practice and demonstrate end-to-end data analysis, SQL, machine learning, and dashboard development.

## Problem
What factors are most associated with airline passenger satisfaction, and how well can we predict it?

## Dataset
Public airline passenger satisfaction dataset (~103,000 records), covering demographics, travel details, delay information, and 14 in-flight service ratings, with a binary satisfaction label.

## Tech Stack
- **Python**: Pandas, NumPy for data cleaning and analysis
- **SQL (SQLite)**: aggregation queries for business-question analysis
- **Scikit-learn**: Logistic Regression (baseline) and Random Forest (final model) for classification
- **Streamlit + Plotly**: interactive dashboard

## Process
1. **Data Cleaning** — handled missing values in arrival delay using median imputation, removed redundant ID columns
2. **Exploratory Analysis** — examined satisfaction across travel class, customer loyalty, and delay levels
3. **SQL Analysis** — wrote aggregation queries (GROUP BY, CASE, ROUND) to quantify satisfaction by segment
4. **Machine Learning** — trained and compared Logistic Regression vs. Random Forest classifiers
5. **Dashboard** — built a 3-tab Streamlit app: Overview KPIs, Segment Analysis, and a live Satisfaction Predictor

## Key Findings
- Business class passengers were **~69% satisfied**, vs. **~19%** for Economy — a ~3.7x gap
- Loyal customers were roughly **2x more likely** to be satisfied than disloyal customers (47.7% vs 23.7%)
- Satisfaction declined with longer delays, but gradually (45.8% → 35.7% as delay increased), not sharply
- **Inflight wifi service, travel class, and type of travel** were the strongest predictors of satisfaction — more influential than delay length itself

## Model Performance
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 85.5% | 0.84 | 0.82 | 0.83 |
| Random Forest | 95.8% | 0.97 | 0.94 | 0.95 |

*Note: feature importance reflects predictive association, not proven causation — these findings would need further investigation before being treated as causal business conclusions.*

## Future Improvements
- Add model explainability (e.g. SHAP values) per individual prediction
- Deploy the dashboard publicly (e.g. Streamlit Community Cloud)
- Add a natural-language insights generator layer

## Author
Kollabathula Joshitha — [GitHub](https://github.com/Joshitha06)
