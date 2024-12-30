import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

st.set_page_config(
    page_title="Retail Foot Traffic Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["Introduction", "Data Simulation", "Exploratory Data Analysis (EDA)",
                                 "Hotspot Analysis", "Foot Traffic Forecasting"])

if app_mode == "Introduction":
    st.title("AI-Powered Retail Foot Traffic")
    st.markdown("""
    Welcome to the Retail Foot Traffic Analysis app! This app demonstrates how AI and data analytics can be applied to retail foot traffic data to provide actionable insights.

    **Features:**
    - Simulate or load foot traffic data.
    - Perform exploratory data analysis.
    - Identify hotspots using clustering algorithms.
    - Forecast future foot traffic using time series models.
    """)
elif app_mode == "Data Simulation":
    st.title("Data Simulation or Upload")
    
    st.header("1. Upload Your Own Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        st.success("Data loaded successfully!")
    else:
        st.header("2. Or Simulate Data")
        if st.button("Simulate Data"):
            # Simulate data
            np.random.seed()
            dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
            foot_traffic = (
                200
                + 50 * np.sin(2 * np.pi * dates.dayofyear / 365)
                + 30 * np.sin(2 * np.pi * dates.dayofweek / 7)
                + np.random.normal(0, 20, len(dates))
            )
            foot_traffic = np.maximum(foot_traffic, 0)
            df = pd.DataFrame({
                'date': dates,
                'foot_traffic': foot_traffic.astype(int)
            })
            st.success("Simulated data created!")
    if 'df' in locals():
        st.write(df.head())
        st.session_state['df'] = df
elif app_mode == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    
    if 'df' not in st.session_state:
        st.warning("Please upload or simulate data first in the 'Data Simulation' section.")
    else:
        df = st.session_state['df']
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['date'], df['foot_traffic'], marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Foot Traffic')
        st.pyplot(fig)
        
        st.subheader("Summary Statistics")
        st.write(df['foot_traffic'].describe())
        
        st.subheader("Distribution Plot")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.histplot(df['foot_traffic'], bins=20, kde=True, ax=ax2)
        st.pyplot(fig2)
elif app_mode == "Hotspot Analysis":
    st.title("Hotspot Analysis")
    
    st.markdown("""
    This section performs clustering on spatial customer data to identify hotspots within the retail space.
    """)
    
    num_customers = st.slider("Number of Customers", 500, 5000, 1000, step=500)
    
    x_base = np.random.uniform(0, 100, size=num_customers)
    y_base = np.random.uniform(0, 100, size=num_customers)
    
    hotspots = [(30, 70), (70, 30), (50, 50)]
    hotspot_customers = []
    for cx, cy in hotspots:
        x_hotspot = np.random.normal(cx, 5, size=200)
        y_hotspot = np.random.normal(cy, 5, size=200)
        hotspot_customers.append((x_hotspot, y_hotspot))
    
    x_coords = np.concatenate([x_base] + [hc[0] for hc in hotspot_customers])
    y_coords = np.concatenate([y_base] + [hc[1] for hc in hotspot_customers])
    positions = pd.DataFrame({'x': x_coords, 'y': y_coords})
    
    st.subheader("Customer Positions")
    fig3 = px.scatter(positions, x='x', y='y', opacity=0.5)
    st.plotly_chart(fig3)
    
    st.subheader("Clustering")
    optimal_k = st.slider("Number of Clusters (k)", 1, 10, 4)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    positions['cluster'] = kmeans.fit_predict(positions[['x', 'y']])
    
    fig4 = px.scatter(positions, x='x', y='y', color=positions['cluster'].astype(str), opacity=0.6)
    st.plotly_chart(fig4)
elif app_mode == "Foot Traffic Forecasting":
    st.title("Foot Traffic Forecasting")
    
    if 'df' not in st.session_state:
        st.warning("Please upload or simulate data first in the 'Data Simulation' section.")
    else:
        df = st.session_state['df']
        periods_input = st.number_input('How many days would you like to forecast into the future?', min_value=1, max_value=365, value=30)
        
        df_prophet = df.rename(columns={'date': 'ds', 'foot_traffic': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        
        with st.spinner('Fitting the model...'):
            model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=periods_input)
        forecast = model.predict(future)
        
        st.subheader('Forecasted Data')
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        st.subheader('Foot Traffic Forecast')
        fig5 = plot_plotly(model, forecast)
        st.plotly_chart(fig5)
        
        st.subheader('Forecast Components')
        fig6 = model.plot_components(forecast)
        st.write(fig6)