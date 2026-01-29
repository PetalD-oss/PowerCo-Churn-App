# PowerCo-Churn-App
An end - to - end machine learning system for predicting customer churn at PowerCo, deployed as an interactive web application using Streamlit and XGBoost.

## OVERVIEW 
This project is a continuation of my previous project (https://github.com/PetalD-oss/PowerCo-Churn-Prediction) by deploying a production-ready churn prediction model. 

## BUSINESS IMPACT 
- **57% Recall**: Successfully identifies over half of at-risk customers before they churn
- **18% Precision**: Strategic balance between catching churners and minimizing false alarms
- **Cost Efficiency**: Customer acquisition costs are 5-10x higher than retention, making even modest precision valuable
- **Revenue Protection**: Early identification enables targeted retention

## KEY FEATURES 
- **Interactive Dashboard**: User-friendly Streamlit interface for real-time predictions
- **Adjustable Risk Threshold**: Customizable sensitivity slider to balance precision vs. recall based on business needs
- **Top Feature Focus**: Uses only the top 7 most important features for ease of use

##  Technical Stack

- **Model**: XGBoost Classifier
- **Deployment**: Streamlit web application
- **Libraries**: scikit-learn, XGBoost, imbalanced-learn, pandas, numpy

## REPO STRUCTURE 
Repository Structure
```
.
├── app.py                      # Streamlit application
├── logistic_model.ipynb        # Model training notebook
├── xgboost_model.pkl          # Trained XGBoost model                
├── feature_names.pkl          # Selected feature list
└── README.md
```
##  Model Development Journey

### Challenge: Severe Class Imbalance
- Only 10% of customers churn (10:1 imbalance ratio)
- Standard models predicted "no churn" for everyone

| Iteration | Model | Technique | Precision | Recall | Key Insight |
|-----------|-------|-----------|-----------|--------|-------------|
| 1 | Logistic Regression | Baseline (4 features) | 13% | 64% | Simple model, high recall |
| 2 | Random Forest | class_weight tuning | 15% | 63% | Ensemble didn't help much |
| 3 | XGBoost | All 24 features | 18% | 50% | Best precision, worse recall |
| 4 | **XGBoost selected feature importance** | **Top 7 features** | **18%** | **57%** | **Best balance** ✅ |

### Final Model Selection
**XGBoost with top feature selection** provided the optimal precision-recall tradeoff while maintaining usability with only 7 input fields instead of 24.

## SETUP 
### PREREQUISITES
- Python 3.8+
- pip package manager

### DEPENDENCIES REQUIRED 
Install dependencies 
- Scikit - learn
- Numpy
- Pandas
- Streamlit
- Matplotlib
- XGBoost 
