import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("Customer Segmentation Dashboard")
st.write("Enter customer details to predict the segment using K-Means Clustering.")

@st.cache_data
def get_trained_model():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Age': np.random.randint(18, 75, size=n),
        'Income': np.random.randint(15000, 120000, size=n),
        'Total_Spending': np.random.randint(100, 3000, size=n),
        'NumWebPurchases': np.random.randint(0, 25, size=n),
        'NumStorePurchases': np.random.randint(0, 20, size=n),
        'NumWebVisitsMonth': np.random.randint(0, 15, size=n),
        'Recency': np.random.randint(0, 100, size=n)
    })
    
    features = ['Age', 'Income', 'Total_Spending', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_
    return kmeans, scaler, df, features

kmeans, scaler, df, features = get_trained_model()

cluster_descriptions = {
    0: "Moderate Spenders: Balanced digital and in-store purchasing pattern.",
    1: "High-Value Shoppers: High income and high spending across categories.",
    2: "Digital-First Shoppers: Frequent website visitors with high online order counts.",
    3: "Budget Conscious: Lower discretionary spending and modest transaction frequency.",
    4: "Dormant / Low-Engagement: High recency intervals indicating inactive accounts.",
    5: "Traditional In-Store Buyers: Higher brick-and-mortar frequency with low digital visits."
}

# Sidebar inputs
st.sidebar.header("Customer Profile Input")
age = st.sidebar.number_input("Age", 18, 100, 35)
income = st.sidebar.number_input("Annual Income ($)", 0, 250000, 50000, step=1000)
total_spending = st.sidebar.number_input("Total Spending ($)", 0, 5000, 1000, step=50)
web_purchases = st.sidebar.number_input("Number of Web Purchases", 0, 100, 10)
store_purchases = st.sidebar.number_input("Number of Store Purchases", 0, 100, 10)
web_visits = st.sidebar.number_input("Number of Web Visits Per Month", 0, 50, 3)
recency = st.sidebar.number_input("Recency (Days Since Last Purchase)", 0, 365, 30)

input_data = pd.DataFrame([{
    'Age': age,
    'Income': income,
    'Total_Spending': total_spending,
    'NumWebPurchases': web_purchases,
    'NumStorePurchases': store_purchases,
    'NumWebVisitsMonth': web_visits,
    'Recency': recency
}])

if st.sidebar.button("Predict Segment"):
    scaled_input = scaler.transform(input_data)
    pred_cluster = int(kmeans.predict(scaled_input)[0])
    
    st.subheader("Predicted Customer Segment")
    st.success(f"Assigned to Cluster {pred_cluster}")
    st.info(f"**Insight:** {cluster_descriptions.get(pred_cluster, 'Standard Segment')}")

st.write("---")
st.subheader("Cluster Distribution & Analysis")
fig = px.scatter(
    df, x='Income', y='Total_Spending', color=df['Cluster'].astype(str),
    title="Customer Clusters: Income vs Total Spending",
    labels={'color': 'Cluster'}
)
st.plotly_chart(fig, use_container_width=True)
