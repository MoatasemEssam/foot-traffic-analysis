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
    uploaded_file = st.file_uploader("Upload a CSV file with 'date' and 'foot_traffic' columns", type=["csv"])

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
    This section performs clustering on spatial customer data to identify hotspots within the retail space. The store layout, including walls, is displayed for context.
    """)

    st.header("1. Upload Your Spatial Data")
    spatial_file = st.file_uploader("Upload a CSV file with 'x' and 'y' coordinates", type=["csv"])

    if spatial_file is not None:
        positions = pd.read_csv(spatial_file)
        if {'x', 'y'}.issubset(positions.columns):
            st.success("Spatial data loaded successfully!")
            st.write(positions.head())
        else:
            st.error("The uploaded CSV file must contain 'x' and 'y' columns.")
            positions = None
    else:
        st.warning("Please upload a CSV file containing 'x' and 'y' coordinates of customer positions.")
        positions = None

    if positions is not None:
        st.header("2. Define Store Layout (Walls)")
        st.markdown("""
        You can either use a predefined store layout or upload a file containing the store's wall coordinates.
        """)

        # Layout Options
        layout_option = st.selectbox("Choose store layout option", ["Default Layout", "Upload Layout File"])

        if layout_option == "Default Layout":
            # Define walls as lines between points
            walls = [
                {'x0': 0, 'y0': 0, 'x1': 100, 'y1': 0},     # Bottom wall
                {'x0': 100, 'y0': 0, 'x1': 100, 'y1': 100},  # Right wall
                {'x0': 100, 'y0': 100, 'x1': 0, 'y1': 100},  # Top wall
                {'x0': 0, 'y0': 100, 'x1': 0, 'y1': 0},     # Left wall
                # Interior wall example
                {'x0': 50, 'y0': 0, 'x1': 50, 'y1': 50},
                {'x0': 50, 'y0': 50, 'x1': 100, 'y1': 50}
            ]
            st.success("Using default store layout.")
        else:
            # Upload layout file
            layout_file = st.file_uploader("Upload a CSV file with wall coordinates ('x0', 'y0', 'x1', 'y1')", type=["csv"])
            if layout_file is not None:
                walls_df = pd.read_csv(layout_file)
                if {'x0', 'y0', 'x1', 'y1'}.issubset(walls_df.columns):
                    walls = walls_df.to_dict('records')
                    st.success("Store layout loaded successfully!")
                    st.write(walls_df.head())
                else:
                    st.error("The uploaded CSV file must contain 'x0', 'y0', 'x1', 'y1' columns.")
                    walls = []
            else:
                st.warning("Please upload a CSV file containing wall coordinates.")
                walls = []

        # Customer Positions with Store Layout
        st.subheader("Customer Positions with Store Layout")
        fig = px.scatter(positions, x='x', y='y', opacity=0.5)

        # Add walls as shapes
        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    x0=wall['x0'], y0=wall['y0'],
                    x1=wall['x1'], y1=wall['y1'],
                    line=dict(color="Black", width=3)
                ) for wall in walls
            ],
            xaxis_title='x',
            yaxis_title='y'
        )
        st.plotly_chart(fig)

        st.subheader("Clustering")

        # Clustering
        optimal_k = st.slider("Number of Clusters (k)", 1, 10, 4)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        positions['cluster'] = kmeans.fit_predict(positions[['x', 'y']])

        fig_clustered = px.scatter(positions, x='x', y='y', color=positions['cluster'].astype(str), opacity=0.6,
                                   labels={'color': 'Cluster'})

        # Add walls to the clustered plot
        fig_clustered.update_layout(
            shapes=[
                dict(
                    type="line",
                    x0=wall['x0'], y0=wall['y0'],
                    x1=wall['x1'], y1=wall['y1'],
                    line=dict(color="Black", width=3)
                ) for wall in walls
            ],
            xaxis_title='x',
            yaxis_title='y'
        )
        st.plotly_chart(fig_clustered)

        st.subheader("Cluster Centers")
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['x', 'y'])
        st.write(cluster_centers)

        # Plot cluster centers with walls and positions
        fig_centers = px.scatter(cluster_centers, x='x', y='y', color_discrete_sequence=['red'], size=[10]*len(cluster_centers), symbol_sequence=['x'])
        fig_positions = px.scatter(positions, x='x', y='y', color=positions['cluster'].astype(str), opacity=0.3)
        for data in fig_positions.data:
            fig_centers.add_trace(data)

        # Add walls to the cluster centers plot
        fig_centers.update_layout(
            shapes=[
                dict(
                    type="line",
                    x0=wall['x0'], y0=wall['y0'],
                    x1=wall['x1'], y1=wall['y1'],
                    line=dict(color="Black", width=3)
                ) for wall in walls
            ],
            xaxis_title='x',
            yaxis_title='y'
        )
        st.plotly_chart(fig_centers)
else:
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