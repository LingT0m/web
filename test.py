import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("insurance_claims.csv")

# Select relevant columns
features = [
    'policy_annual_premium', 'insured_occupation', 'insured_hobbies',
    'capital-gains', 'incident_date', 'incident_severity',
    'bodily_injuries', 'property_claim', 'auto_model'
]
target = 'fraud_reported'

# Preprocess data
df['incident_date'] = pd.to_datetime(df['incident_date'])
df['incident_date'] = df['incident_date'].dt.dayofyear  # Convert date to numerical

le_dict = {}
encoded_df = df.copy()
for col in features:
    if encoded_df[col].dtype == 'object':
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col])
        le_dict[col] = le

# Encode target
target_le = LabelEncoder()
encoded_df[target] = target_le.fit_transform(encoded_df[target])

# Scale numeric features
numeric_features = ['policy_annual_premium', 'capital-gains', 'incident_date', 'bodily_injuries', 'property_claim']
scaler = StandardScaler()
encoded_df[numeric_features] = scaler.fit_transform(encoded_df[numeric_features])

# Train model
X = encoded_df[features]
y = encoded_df[target]
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the model and encoders
joblib.dump(model, "decision_tree_model.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
joblib.dump(target_le, "target_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Streamlit App
st.title("Insurance Claim Fraud Predictor")

st.sidebar.header("Enter Claim Information")

input_data = {}
input_data['policy_annual_premium'] = st.sidebar.number_input("Annual Premium", value=1000.0)
input_data['insured_occupation'] = st.sidebar.selectbox("Occupation", df['insured_occupation'].unique())
input_data['insured_hobbies'] = st.sidebar.selectbox("Hobbies", df['insured_hobbies'].unique())
input_data['capital-gains'] = st.sidebar.number_input("Capital Gains", value=0)
input_data['incident_date'] = st.sidebar.date_input("Incident Date")
input_data['incident_severity'] = st.sidebar.selectbox("Incident Severity", df['incident_severity'].unique())
input_data['bodily_injuries'] = st.sidebar.number_input("Bodily Injuries", value=0)
input_data['property_claim'] = st.sidebar.number_input("Property Claim", value=0)
input_data['auto_model'] = st.sidebar.selectbox("Auto Model", df['auto_model'].unique())

if st.sidebar.button("Predict"):
    # Prepare input
    input_df = pd.DataFrame([input_data])
    input_df['incident_date'] = pd.to_datetime(input_df['incident_date']).dt.dayofyear

    # Load encoders and scaler
    le_dict = joblib.load("label_encoders.pkl")
    target_le = joblib.load("target_encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    for col in input_df.columns:
        if col in le_dict:
            input_df[col] = le_dict[col].transform(input_df[col].astype(str))

    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # Load model
    model = joblib.load("decision_tree_model.pkl")

    # Prediction
    prediction = model.predict(input_df)[0]
    predicted_label = target_le.inverse_transform([prediction])[0]

    st.subheader("Prediction Result")
    st.write(f"The claim is predicted to be: **{predicted_label}**")
