import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go
from prophet.make_holidays import make_holidays_df
import datetime
import io
st.set_page_config(
    page_title="Retail Foot Traffic Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["Introduction", "Data Simulation", "Exploratory Data Analysis (EDA)",
                                 "Hotspot Analysis", "Foot Traffic Forecasting", "Business Insights Report"])

if app_mode == "Introduction":
    st.title("AI-Powered Retail Foot Traffic")
    st.markdown("""
    Welcome to the Retail Foot Traffic Analysis app! This app demonstrates how AI and data analytics can be applied to retail foot traffic data to provide actionable insights.

    **Features:**
    - Simulate or load foot traffic data.
    - Perform exploratory data analysis.
    - Identify hotspots using clustering algorithms.
    - Forecast future foot traffic using time series models.
    - Generate a business insights report.
    """)
elif app_mode == "Data Simulation":
    st.title("Data Simulation or Upload")

    st.header("1. Upload Your Own Data")
    uploaded_file = st.file_uploader("Upload a CSV file with 'date' and 'foot_traffic' columns", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        st.success("Data loaded successfully!")
        st.session_state['df'] = df
    else:
        st.header("2. Or Simulate Data")
        if st.button("Simulate Data"):
            # Simulate data
            np.random.seed(42)
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
            st.session_state['df'] = df

    if 'df' in st.session_state:
        st.write(df.head())
elif app_mode == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")

    if 'df' not in st.session_state:
        st.warning("Please upload or simulate data first in the 'Data Simulation' section.")
    else:
        df = st.session_state['df']

        # Time Series Plot
        st.subheader("Foot Traffic Over Time")
        fig = px.line(df, x='date', y='foot_traffic', title='Daily Foot Traffic', labels={'foot_traffic': 'Foot Traffic', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)

        # Summary Statistics
        st.subheader("Summary Statistics")
        stats = df['foot_traffic'].describe()
        st.table(stats)

        # Distribution Plot
        st.subheader("Distribution of Foot Traffic")
        fig2 = px.histogram(df, x='foot_traffic', nbins=20, title='Distribution of Foot Traffic', labels={'foot_traffic': 'Foot Traffic'})
        st.plotly_chart(fig2, use_container_width=True)

        # Additional Analysis: Traffic by Day of Week
        st.subheader("Average Foot Traffic by Day of Week")
        df['day_of_week'] = df['date'].dt.day_name()
        avg_traffic_weekday = df.groupby('day_of_week')['foot_traffic'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig3 = px.bar(x=avg_traffic_weekday.index, y=avg_traffic_weekday.values, title='Average Foot Traffic by Day of Week', labels={'x': 'Day of Week', 'y': 'Average Foot Traffic'})
        st.plotly_chart(fig3, use_container_width=True)

elif app_mode == "Hotspot Analysis":
    st.title("Hotspot Analysis")

    st.markdown("""
    This section performs clustering on spatial customer data to identify hotspots within the retail space. The store layout, including walls, is displayed for context.
    """)

    st.header("1. Choose How to Input Spatial Data")

    data_option = st.selectbox("Select data input method", ["Upload Data", "Simulate Data", "Load Data from Camera"])

    if data_option == "Upload Data":
        st.subheader("Upload Customer Positions")
        uploaded_file = st.file_uploader("Upload a CSV file with 'x' and 'y' columns", type=["csv"])
        if uploaded_file is not None:
            positions = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            st.write(positions.head())
        else:
            st.warning("Please upload a CSV file.")
    elif data_option == "Simulate Data":
        st.subheader("Simulate Customer Positions")
        num_customers = st.slider("Number of customers to simulate", 10, 500, 100)
        positions = pd.DataFrame({
            'x': np.random.uniform(0, 100, num_customers),
            'y': np.random.uniform(0, 100, num_customers)
        })
        st.success("Simulated customer positions created!")
        st.write(positions.head())
    else:  # Load Data from Camera
        st.subheader("Load Customer Positions from Camera Data")
        st.markdown("Data will be loaded from the `positions.csv` file generated by the external data processing script.")

        if st.button("Refresh Data"):
            try:
                positions = pd.read_csv('positions.csv')
                if not positions.empty:
                    st.success("Customer positions loaded successfully!")
                    st.write(positions.head())
                else:
                    st.warning("No customer positions found in `positions.csv`.")
                    positions = None
            except FileNotFoundError:
                st.error("`positions.csv` file not found. Please ensure the external data processing script is running.")
                positions = None
        else:
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
        fig = go.Figure()

        # Plot customer positions
        fig.add_trace(go.Scatter(
            x=positions['x'], y=positions['y'],
            mode='markers',
            name='Customers',
            marker=dict(color='blue', size=5, opacity=0.5)
        ))

        # Add walls as shapes
        for wall in walls:
            fig.add_shape(
                type="line",
                x0=wall['x0'], y0=wall['y0'],
                x1=wall['x1'], y1=wall['y1'],
                line=dict(color="black", width=3),
            )

        fig.update_layout(
            title='Customer Positions with Store Layout',
            xaxis_title='X Position',
            yaxis_title='Y Position',
            showlegend=False,
            width=800, height=600
        )
        st.plotly_chart(fig)

        st.subheader("Clustering and Hotspot Identification")

        # Clustering
        optimal_k = st.slider("Number of Clusters (k)", 1, 10, 4)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        positions['cluster'] = kmeans.fit_predict(positions[['x', 'y']])

        # Plot clustered positions
        fig_clustered = px.scatter(positions, x='x', y='y', color=positions['cluster'].astype(str),
                                   title='Customer Clusters',
                                   labels={'x': 'X Position', 'y': 'Y Position', 'color': 'Cluster'})
        # Add walls to the plot
        for wall in walls:
            fig_clustered.add_shape(
                type="line",
                x0=wall['x0'], y0=wall['y0'],
                x1=wall['x1'], y1=wall['y1'],
                line=dict(color="black", width=3),
            )
        fig_clustered.update_layout(width=800, height=600)
        st.plotly_chart(fig_clustered)

        st.subheader("Cluster Analysis")

        # Cluster Centers
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['x', 'y'])
        cluster_counts = positions['cluster'].value_counts().sort_index()
        cluster_summary = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Number of Customers': cluster_counts.values,
            'Center X': cluster_centers['x'],
            'Center Y': cluster_centers['y']
        })
        st.table(cluster_summary)

        st.markdown("""
        **Insights:**

        - The clusters represent areas in the store where customers tend to gather.
        - Cluster centers indicate the hotspots within the store layout.
        - Higher customer counts in a cluster suggest more popular areas.
        """)
# ... after clustering and plotting are done
# Store data in session_state
    st.session_state['positions'] = positions
    st.session_state['cluster_summary'] = cluster_summary
    st.session_state['fig_clustered'] = fig_clustered

elif app_mode == "Foot Traffic Forecasting":
    st.title("Foot Traffic Forecasting")

    if 'df' not in st.session_state:
        st.warning("Please upload or simulate data first in the 'Data Simulation' section.")
    else:
        df = st.session_state['df']

        # Select forecast parameters
        st.subheader("Forecast Parameters")
        periods_input = st.number_input('Number of days to forecast into the future:', min_value=1, max_value=365, value=30)
        st.write("")

        # Option to include holidays
        st.subheader("Include Holidays in the Model")
        holiday_option = st.radio(
            "Do you want to include holidays in the model?",
            ('No', 'Use Predefined Holidays', 'Input Custom Holidays', 'Upload Holiday File'))

        if holiday_option == 'Use Predefined Holidays':
            # Define a dataframe of predefined holidays (example using US holidays)
            from prophet.make_holidays import make_holidays_df
            years = [df['date'].dt.year.min(), df['date'].dt.year.max() + 1]
            holidays = make_holidays_df(year_list=list(range(years[0], years[1] + 1)), country='US')
            st.success("Predefined holidays included in the model.")

        elif holiday_option == 'Input Custom Holidays':
            st.markdown("### Input Custom Holiday Dates and Names")
            num_holidays = st.number_input('How many holidays do you want to input?', min_value=1, max_value=20, value=5)
            holiday_dates = []
            for i in range(num_holidays):
                cols = st.columns(2)
                with cols[0]:
                    date = st.date_input(f'Holiday Date {i+1}', value=None, key=f'holiday_date_{i}')
                with cols[1]:
                    name = st.text_input(f'Holiday Name {i+1}', value=f'Holiday_{i+1}', key=f'holiday_name_{i}')
                if date:
                    holiday_dates.append({'ds': pd.to_datetime(date), 'holiday': name})
            if holiday_dates:
                holidays = pd.DataFrame(holiday_dates)
                st.success(f"{len(holiday_dates)} custom holidays added.")
                st.write(holidays)
            else:
                st.warning("No holidays have been added.")
                holidays = None

        elif holiday_option == 'Upload Holiday File':
            st.markdown("### Upload a CSV File with Holiday Dates")
            uploaded_holiday_file = st.file_uploader("Upload a CSV file with 'ds' and 'holiday' columns", type=["csv"])
            if uploaded_holiday_file is not None:
                holidays = pd.read_csv(uploaded_holiday_file, parse_dates=['ds'])
                if 'holiday' not in holidays.columns:
                    holidays['holiday'] = 'custom_holiday'
                st.success("Holidays from the uploaded file included.")
                st.write(holidays)
            else:
                st.warning("Please upload a CSV file with holiday dates.")
                holidays = None
        else:
            holidays = None

        # Split data into training and testing
        st.subheader("Model Training and Evaluation")
        test_size = st.slider('Number of days in test set:', min_value=7, max_value=90, value=30, step=7)
        df_train = df.iloc[:-test_size]
        df_test = df.iloc[-test_size:]

        # Prepare data for Prophet
        df_train_prophet = df_train.rename(columns={'date': 'ds', 'foot_traffic': 'y'})
        df_test_prophet = df_test.rename(columns={'date': 'ds', 'foot_traffic': 'y'})

        # Initialize and train the model
        with st.spinner('Training the forecasting model...'):
            model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_train_prophet)

        # Prepare future dataframe
        future = model.make_future_dataframe(periods=test_size + periods_input)
        forecast = model.predict(future)

        # Evaluate model performance
        st.subheader("Model Performance on Test Data")
        forecast_test = forecast[forecast['ds'].isin(df_test_prophet['ds'])]
        forecast_test = forecast_test.merge(df_test_prophet[['ds', 'y']], on='ds')
        forecast_test['error'] = forecast_test['y'] - forecast_test['yhat']
        forecast_test['abs_error'] = forecast_test['error'].abs()
        mape = np.mean(forecast_test['abs_error']/forecast_test['y'])*100
        rmse = np.sqrt(np.mean(forecast_test['error']**2))

        st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

        # Plot actual vs. predicted
        st.subheader("Actual vs. Predicted Foot Traffic")
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=forecast_test['ds'],
            y=forecast_test['y'],
            mode='lines+markers',
            name='Actual'
        ))
        fig_compare.add_trace(go.Scatter(
            x=forecast_test['ds'],
            y=forecast_test['yhat'],
            mode='lines+markers',
            name='Predicted'
        ))
        fig_compare.update_layout(
            title='Actual vs. Predicted Foot Traffic',
            xaxis_title='Date',
            yaxis_title='Foot Traffic'
        )
        st.plotly_chart(fig_compare)

        # Forecast future
        st.subheader("Future Forecast")
        forecast_future = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-periods_input:].copy()
        forecast_future = forecast_future.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Foot Traffic', 
                                                          'yhat_lower': 'Lower Confidence Interval', 
                                                          'yhat_upper': 'Upper Confidence Interval'})
        st.write(forecast_future)

        # Plot future forecast with confidence intervals
        fig_future = go.Figure([
            go.Scatter(
                name='Upper Confidence Interval',
                x=forecast_future['Date'],
                y=forecast_future['Upper Confidence Interval'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Forecasted Foot Traffic',
                x=forecast_future['Date'],
                y=forecast_future['Forecasted Foot Traffic'],
                mode='lines',
                line=dict(color='blue'),
                fill='tonexty',
                fillcolor='rgba(68, 68, 68, 0.1)',
            ),
            go.Scatter(
                name='Lower Confidence Interval',
                x=forecast_future['Date'],
                y=forecast_future['Lower Confidence Interval'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(68, 68, 68, 0.1)',
                showlegend=False
            )
        ])
        fig_future.update_layout(
            title='Foot Traffic Forecast',
            xaxis_title='Date',
            yaxis_title='Foot Traffic',
            showlegend=True
        )
        st.plotly_chart(fig_future)

        # Plot forecast components
        # Plot forecast components without holidays
        # Plot forecast components without holidays
# Plot forecast components
        st.subheader('Forecast Components')

        components = ['trend', 'weekly']  # Add 'yearly' if desired

        fig = model.plot_components(forecast)  # This will include all components

        st.pyplot(fig)

        st.markdown("""
        **Insights:**

        - The model's performance on the test data indicates its ability to generalize to unseen data.
        - Including holidays may improve the model's accuracy if holidays significantly impact foot traffic.
        - The components plot reveals patterns in the data, such as trends, weekly seasonality, and the effect of holidays.
        """)

        # Store variables in session_state
        st.session_state['forecast'] = forecast
        st.session_state['forecast_future'] = forecast_future
        st.session_state['fig_future'] = fig_future
        st.session_state['periods_input'] = periods_input   

# Business Insights Report Section
elif app_mode == "Business Insights Report":
    st.title("Business Insights Report")

    if 'df' not in st.session_state:
        st.warning("Please complete the previous sections to generate the report.")
    else:
        df = st.session_state['df']
        # Ensure 'date' column is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Function to generate the report HTML
        def generate_report_html():
            import io
            from datetime import datetime

            report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Initialize an HTML content string
            html_parts = []
            html_parts.append(f"""
            <html>
            <head>
                <title>Business Insights Report</title>
                <style>
                    /* Include your CSS styles here */
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2, h3 {{ color: #2E4053; }}
                    p {{ font-size: 14px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    ul {{ list-style-type: disc; margin-left: 20px; }}
                </style>
            </head>
            <body>
            <h1>Business Insights Report</h1>
            <p><strong>Generated on:</strong> {report_date}</p>
            """)

            # Executive Summary
            html_parts.append("""
            <h2>Executive Summary</h2>
            <p>The following report provides insights into the foot traffic patterns and customer behavior within your retail space...</p>
            """)

            # Foot Traffic Over Time Section
            html_parts.append("<h2>Foot Traffic Over Time</h2>")
            try:
                # Generate the figure
                fig = px.line(df, x='date', y='foot_traffic', labels={'foot_traffic': 'Foot Traffic', 'date': 'Date'})
                fig_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
                html_parts.append(fig_html)

                # Average Foot Traffic by Day of Week Section
                # Ensure 'date' column is datetime
                df['date'] = pd.to_datetime(df['date'])

                # Calculate day of the week
                df['day_of_week'] = df['date'].dt.day_name()

                # Calculate average foot traffic by day of the week
                avg_traffic_weekday = df.groupby('day_of_week')['foot_traffic'].mean().reindex([
                    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

                # Generate the figure
                fig3 = px.bar(x=avg_traffic_weekday.index, y=avg_traffic_weekday.values,
                              labels={'x': 'Day of Week', 'y': 'Average Foot Traffic'})

                # Convert the figure to HTML
                fig3_html = fig3.to_html(include_plotlyjs='cdn', full_html=False)

                html_parts.append("<h3>Average Foot Traffic by Day of Week</h3>")
                html_parts.append(fig3_html)

                ### Compute Key Observations ###
                # Find the day(s) with the highest average foot traffic
                max_avg = avg_traffic_weekday.max()
                peak_days = avg_traffic_weekday[avg_traffic_weekday == max_avg].index.tolist()
                peak_days_str = ', '.join(peak_days)

                # Find the busiest day
                peak_date = df[df['foot_traffic'] == df['foot_traffic'].max()]['date'].dt.strftime('%Y-%m-%d').values[0]

                html_parts.append(f"""
                <h3>Key Observations</h3>
                <ul>
                    <li>Foot traffic shows a clear weekly pattern with peaks on <strong>{peak_days_str}</strong>.</li>
                    <li>The busiest day was <strong>{peak_date}</strong>, suggesting optimal times for promotions or increased staffing.</li>
                </ul>
                """)
            except Exception as e:
                html_parts.append(f"<p><em>An error occurred while processing the 'Foot Traffic Over Time' section: {e}</em></p>")

            # Hotspot Analysis
            html_parts.append("<h2>Hotspot Analysis</h2>")

            if 'positions' in st.session_state and 'cluster_summary' in st.session_state and 'fig_clustered' in st.session_state:
                positions = st.session_state['positions']
                cluster_summary = st.session_state['cluster_summary']
                fig_clustered = st.session_state['fig_clustered']

                # Convert the clustered plot to HTML
                fig_clustered_html = fig_clustered.to_html(include_plotlyjs='cdn', full_html=False)
                html_parts.append("<h3>Customer Clusters</h3>")
                html_parts.append(fig_clustered_html)

                html_parts.append("<h3>Cluster Summary</h3>")
                # Convert cluster summary to HTML table
                cluster_summary_html = cluster_summary.to_html(index=False)
                html_parts.append(cluster_summary_html)

                # Key Observations
                html_parts.append("""
                <h3>Key Observations</h3>
                <ul>
                """)
                for index, row in cluster_summary.iterrows():
                    html_parts.append(f"<li><strong>Cluster {row['Cluster']}:</strong> Center at ({row['Center X']:.2f}, {row['Center Y']:.2f}), with {row['Number of Customers']} customers.</li>")
                html_parts.append("""
                </ul>
                <p>These clusters indicate areas where customers tend to spend more time. Consider placing high-margin products or promotional displays in these hotspots to maximize sales.</p>
                """)
            else:
                html_parts.append("<p><em>Hotspot analysis data not available. Please complete the Hotspot Analysis section.</em></p>")

            # Foot Traffic Forecast
            html_parts.append("<h2>Foot Traffic Forecast</h2>")

            if 'forecast_future' in st.session_state and 'fig_future' in st.session_state:
                forecast_future = st.session_state['forecast_future']
                fig_future = st.session_state['fig_future']
                periods_input = st.session_state.get('periods_input', 30)

                # Convert the forecast plot to HTML
                fig_future_html = fig_future.to_html(include_plotlyjs='cdn', full_html=False)
                html_parts.append("<h3>Forecasted Foot Traffic</h3>")
                html_parts.append(fig_future_html)

                # Convert forecast data to HTML table
                forecast_future_html = forecast_future.to_html(index=False)
                html_parts.append("<h3>Forecast Data</h3>")
                html_parts.append(forecast_future_html)

                # Compute Key Observations
                recent_past_avg = df['foot_traffic'].tail(periods_input).mean()
                forecast_avg = forecast_future['Forecasted Foot Traffic'].mean()

                if forecast_avg > recent_past_avg:
                    trend = 'increase'
                else:
                    trend = 'decrease'

                top_future_dates = forecast_future.nlargest(3, 'Forecasted Foot Traffic')['Date'].dt.strftime('%Y-%m-%d').tolist()
                top_dates_str = ', '.join(top_future_dates)

                html_parts.append(f"""
                <h3>Key Observations</h3>
                <ul>
                    <li>Foot traffic is expected to <strong>{trend}</strong> over the next <strong>{periods_input}</strong> days.</li>
                    <li>Anticipate higher traffic on <strong>{top_dates_str}</strong>, allowing for proactive planning.</li>
                </ul>
                """)
            else:
                html_parts.append("<p><em>Forecast data not available. Please complete the Foot Traffic Forecasting section.</em></p>")

            # Recommendations
            html_parts.append("""
            <h2>Recommendations</h2>
            <ul>
                <li><strong>Staffing Adjustments:</strong> Align staff schedules with peak traffic times to ensure optimal customer service.</li>
                <li><strong>Promotional Timing:</strong> Schedule marketing promotions during periods of increased foot traffic for maximum impact.</li>
                <li><strong>Store Layout Optimization:</strong> Place popular or high-margin items in identified hotspots to boost sales.</li>
                <li><strong>Future Planning:</strong> Use the forecasted trends to plan inventory levels and seasonal promotions.</li>
            </ul>
            """)

            # Conclusion
            html_parts.append("""
            <h2>Conclusion</h2>
            <p>Leveraging data analytics provides valuable insights into customer behavior and foot traffic patterns. By integrating these findings into your operational and strategic planning, you can enhance customer experience, increase sales, and improve overall store performance.</p>
            """)

            # End of HTML content
            html_parts.append("""
            </body>
            </html>
            """)

            # Combine all parts into a single HTML
            html_report = ''.join(html_parts)

            return html_report

        # Generate the report HTML
        html_report = generate_report_html()

        # Display the report in the app
        st.components.v1.html(html_report, height=800, scrolling=True)

        # Provide a download button
        st.subheader("Download Report")
        if st.button("Generate and Download Report"):
            # The report is already generated, so we can use 'html_report'
            st.download_button(
                label="Download Report as HTML",
                data=html_report,
                file_name='business_insights_report.html',
                mime='text/html'
            )
       

        