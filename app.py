import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
st.set_page_config(page_title="Customer Lifetime Value (CLV) Predictor",layout="wide")
st.title("Customer Lifetime Value (CLV) Prediction App")
st.sidebar.header("Upload Excel Dataset")
uploaded_file = st.sidebar.file_uploader("Choose the `online_retail_II.xlsx` file",type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file,sheet_name='Year 2010-2011')
    st.subheader("🔍 Data Overview")
    st.write("Shape of dataset:",data.shape)
    st.dataframe(data.head())
    df=data.dropna(subset=["Customer ID"])
    df=df[df["Quantity"] > 0]
    df["TotalPrice"]=df["Quantity"]*df["Price"]
    df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"])
    st.subheader("🧮 RFM Feature Engineering")
    snapshot_date=df["InvoiceDate"].max()+timedelta(days=1)

    rfm=df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "TotalPrice": "sum"
    })
    rfm.columns=["Recency","Frequency","Monetary"]
    rfm=rfm[rfm["Monetary"]>0]

    st.write("Top customers based on RFM:")
    st.dataframe(rfm.sort_values("Monetary",ascending=False).head(10))
    st.subheader("📈 RFM Distributions")
    col1,col2,col3=st.columns(3)
    with col1:
        st.plotly_chart(px.histogram(rfm,x="Recency",nbins=30,title="Recency Distribution"),use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(rfm,x="Frequency",nbins=30,title="Frequency Distribution"),use_container_width=True)
    with col3:
        st.plotly_chart(px.histogram(rfm,x="Monetary",nbins=30,title="Monetary Value Distribution"),use_container_width=True)
    st.subheader("🤖 CLV Model Training")
    X = rfm[["Recency","Frequency"]]
    y = rfm["Monetary"]
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    st.write("Model Performance:")
    st.write(f"MAE:{mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"R² Score:{r2_score(y_test, y_pred):.2f}")
    st.subheader("🔮 Predict Customer CLV")
    recency_input=st.slider("Recency (days since last purchase)",0,365,90)
    frequency_input=st.slider("Frequency (number of purchases)",1,100,10)
    clv_prediction=model.predict([[recency_input, frequency_input]])[0]
    st.success(f"💰 Predicted CLV: £{clv_prediction:.2f}")
    st.subheader("🎯 Customer Segmentation")
    rfm["Segment"]=pd.qcut(rfm["Monetary"],4,labels=["Low","Mid-Low","Mid-High","High"])
    seg_counts=rfm["Segment"].value_counts().sort_index()
    st.plotly_chart(px.pie(values=seg_counts.values,names=seg_counts.index,title="Customer Segments"),use_container_width=True)
    st.markdown("---")

else:
    st.warning("👈 Please upload the dataset to get started.")
