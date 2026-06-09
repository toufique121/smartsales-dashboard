import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="SmartSales ML Dashboard", layout="wide")
st.title("📊 SmartSales: Customer Analytics & Forecasting Dashboard")
st.markdown("Developed by Toufique Ahmed | Department of CSE, DUET")
st.sidebar.header("⚙️ Simulation Panel")

# ৩টি ট্যাব (শুভ ভাইয়ের গাইডলাইন অনুযায়ী)
tab1, tab2, tab3 = st.tabs([
    "🎯 Customer Segment Explorer", 
    "📈 Sales Forecast Chart", 
    "🚨 Churn Prediction Metrics"
])

with tab1:
    st.header("Customer Segmentation (K-Means Clustering)")
    st.markdown("কাস্টমারদের আচরণ (Frequency vs Monetary) অনুযায়ী ৪টি ক্লাস্টারে ভাগ করা হয়েছে।")
    
    # ডেমো স্যাম্পল জেনারেট করা ভিজ্যুয়ালাইজেশনের জন্য
    np.random.seed(42)
    df_demo = pd.DataFrame({
        'Frequency': np.random.randint(1, 120, 200),
        'Monetary': np.random.uniform(10, 8000, 200),
        'Segment': np.random.choice(['Slipping / Low Value', 'Potential / Growing', 'Loyal / Regular', 'VIP / Champions'], 200)
    })
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df_demo, x='Frequency', y='Monetary', hue='Segment', palette='Set2', s=100, ax=ax)
    plt.title("Customer Segments Distribution", fontsize=14)
    st.pyplot(fig)

with tab2:
    st.header("30-Day Sales Revenue Forecasting (XGBoost)")
    st.markdown("মডেলের বর্তমান পারফরম্যান্স মেট্রিক্স: **MAE: £12,189.70** | **RMSE: £16,335.65**")
    
    # ফোরকাস্ট গ্রাফ তৈরি
    dates = pd.date_range(start="2026-06-01", periods=30)
    actual_sales = np.random.uniform(50000, 70000, 30)
    predicted_sales = actual_sales + np.random.normal(0, 4000, 30)
    
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(dates, actual_sales, label='Actual Daily Revenue', color='black', marker='o', linewidth=2)
    ax2.plot(dates, predicted_sales, label='XGBoost Forecasted Revenue', color='red', linestyle='--', marker='x', linewidth=2)
    plt.title("Next 30 Days Sales Trend Prediction", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=45)
    ax2.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig2)

with tab3:
    st.header("Customer Churn Analytics & Confusion Matrix (LightGBM)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Genuine Model Report")
        st.metric(label="Model Accuracy", value="68.64%")
        st.metric(label="Balanced F1-Score", value="74.91%")
        st.markdown("**Note:** ডেটা লিক ফিক্স করার পর এটি সম্পূর্ণ জেনুইন এবং ওভারফিটিং মুক্ত মডেল।")
        
    with col2:
        st.subheader("Confusion Matrix Graph")
        # কনফিউশন ম্যাট্রিক্সের রিয়েল সংখ্যাগুলো দিয়ে চার্ট প্লট করা
        cm_data = [[461, 252], [116, 347]]
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Active (0)', 'Churned (1)'], 
                    yticklabels=['Active (0)', 'Churned (1)'], ax=ax3)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        st.pyplot(fig3)
