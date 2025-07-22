import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="🧮 CLV Predictor", layout="wide")

st.title("📊 Customer Lifetime Value Prediction App")
st.markdown("Estimate how much value a customer brings over time using **RFM features** and ML.")

# ----------------------
# Helper: Train Model
# ----------------------
@st.cache_resource
def train_model_from_real_data():
    df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=["CustomerID"], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    snapshot_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]

    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Monetary'] * 1.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    joblib.dump(model, "clv_model.pkl")
    return model, rfm, rmse

# Load model and RFM data
if not os.path.exists("clv_model.pkl"):
    model, rfm_data, rmse = train_model_from_real_data()
else:
    model = joblib.load("clv_model.pkl")
    rfm_data = pd.read_csv("rfm_backup.csv")
    rmse = None

# Sidebar for input
st.sidebar.header("Upload CSV")
uploaded = st.sidebar.file_uploader("Upload CSV with Recency, Frequency, Monetary", type=["csv"])

# Main UI display
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📈 RFM Distribution")
    fig1 = px.scatter(rfm_data, x="Recency", y="Monetary", size="Frequency", color="Monetary",
                      title="RFM Customer Segments", height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.metric("Model RMSE", f"{rmse:.2f}" if rmse else "Loaded", delta="Lower is better")

# Prediction area
st.subheader("🔮 Predict CLV")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.write("Uploaded Data:", df.head())
        predictions = model.predict(df[['Recency', 'Frequency', 'Monetary']])
        df['Predicted_CLV'] = predictions
        st.success("Prediction successful!")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "clv_results.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV or use our trained model preview above.")

st.markdown("---")
st.caption("Built with ❤️ using real e-commerce data.")