import streamlit as st 
import pickle 
import numpy as np 
from xgboost import XGBClassifier 

@st.cache_resource
def load_model(): 
     model = pickle.load(open('xg_boost_model.pkl', 'rb'))
     features = pickle.load(open('feature_names.pkl', 'rb'))
     return model, features

model, features = load_model()

st.title("How likely is the Customer to Churn?")

# XG Model Stats with 6 features
st.subheader("Model Stats")
col1, col2, col3 = st.columns(3)
with col1: 
     st.metric("Precision", "18%")
with col2: 
     st.metric("Recall", "57%")
with col3: 
     st.metric("Accuracy", "70%")

st.divider()
st.subheader("How Strict should we be?")
churn_threshold = st.slider("Slide for threshold of your choice",0.0, 1.0, 0.5, 0.05)

st.write("Slider value is:", churn_threshold)
st.caption("Note for user: Confused what to choose?  \n" 
"**Lower Values** will flag more customers as high risk  \n"
"While **Higher Values** will lead to catching less customers but with an accurate precision")

st.subheader("Enter the following details", divider="blue")
margin_net_pow_ele = st.number_input('Net Margin on Power Subscription', value=0.0)
margin_gross_pow_ele = st.number_input('Gross Margin on Power Subscription', value = 0.0)
cons_12m = st.number_input('Electricity consumptions over 12 month period', value = 0.0)
num_years_antig = st.number_input('Customer Tenure', value = 0.0)
forecast_cons_12m = st.number_input('Forecasted Electricity consumption for the next 12 months', value = 0.0)
forecast_meter_rent_12m = st.number_input('Forecasted bill of meter rental for the next 2 months', value=0.0)

if st.button('Predict Churn'):
     input_data = np.array([[float(margin_gross_pow_ele),float (margin_net_pow_ele), float (cons_12m), float (num_years_antig), float(forecast_cons_12m), float(forecast_meter_rent_12m)]])
     probability = model.predict_proba(input_data)
     churn_prob = probability[0][1]

     st.divider()
     col1, col2 = st.columns(2)
     with col1: 
          st.metric("Churn Probability (Model Output)",f"{churn_prob:.1%}", 
                    delta=f"{(churn_prob - churn_threshold):.1%}" if churn_prob > churn_threshold else None,
                   delta_color="inverse")
     with col2: 
          st.metric("Retention Likelihood",f"{probability[0][0]:.1%}")
     
     if churn_prob >= churn_threshold:
          st.error('**HIGH CHURN RISK** - Recommend retention action')
          st.write("**Suggested Actions:**")
          st.write("- Reach out with personalized offer")
          st.write("- Review account for service issues")
          st.write("- Consider loyalty discount")
     else:
          st.success('**LOW CHURN RISK** - Customer likely to stay')
          
     
 #model's confidence 
     confidence = "High" if max(probability[0]) > 0.7 else "Medium" if max(probability[0]) > 0.5 else "Low"
     st.info(f"Model Confidence: **{confidence}**")