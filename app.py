import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("💳Fraud Detection App")

uploaded_file = st.file_uploader("📁 Upload a small CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Class" in df.columns:
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        X = df.drop("Class", axis=1)
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X_train, y_train)

        df["Prediction"] = model.predict(X)
        frauds = df[df["Prediction"] == 1]

        st.success(f"✅ Total Transactions: {len(df)}")
        st.error(f"🚨 Frauds Detected: {len(frauds)}")
        st.dataframe(frauds)
    else:
        st.warning("Please upload a CSV with a 'Class' column.")
