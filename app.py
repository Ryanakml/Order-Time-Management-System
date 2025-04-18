import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os

# Import model modules
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Import preprocessing functions
from preprocessing import preprocess_data

# Set page config
st.set_page_config(page_title="Order Prediction", layout="wide")

# App title
st.title("Order Prediction System")
st.markdown("App Prediction order berbasis time-series")

# Sidebar
st.sidebar.header("Settings")

# Data options
data_option = st.sidebar.radio(
    "Data Source",
    ["Upload Data", "Data Management"]
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False
if 'edited_data' not in st.session_state:
    st.session_state.edited_data = None

# Function to save data to CSV
def save_data_to_csv(data, filename="order_data.csv"):
    data.to_csv(filename, index=False)
    return filename

# Function to load data from CSV
def load_data_from_csv(filename="order_data.csv"):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

# Data loading section
st.header("Data")

# Data Management Section
elif data_option == "Data Management":
    st.subheader("Data Management")
    
    # Initialize edited data if not already done
    if st.session_state.edited_data is None:
        if st.session_state.data is not None:
            st.session_state.edited_data = st.session_state.data.copy()
        else:
            # Try to load from file
            loaded_data = load_data_from_csv()
            if loaded_data is not None:
                st.session_state.edited_data = loaded_data
                st.session_state.data = loaded_data.copy()
            else:
                # Create empty DataFrame with required columns
                empty_df = pd.DataFrame(columns=['timestamp', 'jumlah_order', 'lokasi'])
                st.session_state.edited_data = empty_df
                st.session_state.data = empty_df.copy()
                st.info("No existing data found. You can add new data below.")
    
    if st.session_state.edited_data is not None:
        # Make a copy to avoid modifying the original during the session
        working_data = st.session_state.edited_data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in working_data.columns:
            working_data['timestamp'] = pd.to_datetime(working_data['timestamp'])
        
        # Add new data form
        st.subheader("Add New Data")
        with st.form("add_data_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_timestamp = st.date_input("Date", value=datetime.now())
                new_hour = st.selectbox("Hour", list(range(24)), index=datetime.now().hour)
            with col2:
                new_order_count = st.number_input("Order Count", min_value=0, value=50)
            with col3:
                locations = ['Jakarta Pusat', 'Jakarta Barat', 'Jakarta Selatan', 
                            'Jakarta Timur', 'Jakarta Utara', 'Depok', 'Tangerang', 'Bekasi']
                new_location = st.selectbox("Location", locations)
            
            submit_button = st.form_submit_button("Add Data")
            
            if submit_button:
                # Create timestamp with selected date and hour
                full_timestamp = datetime.combine(new_timestamp, datetime.min.time()) + timedelta(hours=new_hour)
                
                # Add new row to dataframe
                new_row = pd.DataFrame({
                    'timestamp': [full_timestamp],
                    'jumlah_order': [new_order_count],
                    'lokasi': [new_location]
                })
                
                # Append to existing data
                st.session_state.edited_data = pd.concat([st.session_state.edited_data, new_row], ignore_index=True)
                st.success("Data added successfully!")
                
                # Update working data
                working_data = st.session_state.edited_data.copy()
                working_data['timestamp'] = pd.to_datetime(working_data['timestamp'])
        
        # Display and edit existing data
        st.subheader("View and Edit Data")
        
        # Sort data by timestamp for better viewing
        working_data = working_data.sort_values(by='timestamp', ascending=False)
        
        # Create a dataframe editor
        edited_df = st.data_editor(
            working_data,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm",
                    step=3600,  # 1 hour in seconds
                ),
                "jumlah_order": st.column_config.NumberColumn(
                    "Order Count",
                    min_value=0,
                    format="%d"
                ),
                "lokasi": st.column_config.SelectboxColumn(
                    "Location",
                    options=locations,
                    width="medium",
                )
            }
        )
        
        # Save button for edited data
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes"):
                st.session_state.edited_data = edited_df.copy()
                st.session_state.data = edited_df.copy()  # Update main data as well
                filename = save_data_to_csv(edited_df)
                st.success(f"Data saved successfully to {filename}!")
        
        with col2:
            if st.button("Train Model with Updated Data"):
                if st.session_state.edited_data is not None and not st.session_state.edited_data.empty:
                    st.session_state.data = st.session_state.edited_data.copy()
                    st.session_state.is_trained = False  # Reset training flag
                    st.success("Data updated for model training. Please go to prediction tab and train the model.")
                else:
                    st.error("No data available for training.")

# Upload Data Section
if data_option == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            # Check if required columns exist
            if 'timestamp' in data.columns and 'jumlah_order' in data.columns:
                # Convert timestamp to datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                st.session_state.data = data
                st.session_state.edited_data = data.copy()  # Copy to edited data
                st.success(f"Loaded {len(data)} records from uploaded file")
            else:
                st.error("Uploaded file must contain 'timestamp' and 'jumlah_order' columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Display data if available
if st.session_state.data is not None and len(st.session_state.data) > 0 and data_option != "Data Management":
    # Date range filter
    min_date = st.session_state.data['timestamp'].min().date()
    max_date = st.session_state.data['timestamp'].max().date()
    
    date_range = st.date_input(
        "Select date range for analysis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = st.session_state.data[
            (st.session_state.data['timestamp'].dt.date >= start_date) &
            (st.session_state.data['timestamp'].dt.date <= end_date)
        ]
    else:
        filtered_data = st.session_state.data
    
    # Display data table
    with st.expander("View Data"):
        st.dataframe(filtered_data)
    
    # Data visualization
    st.subheader("Historical Order Data")
    
    # Aggregation options
    agg_option = st.selectbox(
        "Aggregation Level",
        ["Hourly", "Daily", "Weekly"]
    )
    
    # Preprocess and aggregate data
    processed_data = preprocess_data(filtered_data, agg_level=agg_option.lower())
    
    # Plot historical data
    fig = px.line(
        processed_data, 
        x='ds', 
        y='y', 
        title=f"{agg_option} Order Data",
        labels={'ds': 'Date', 'y': 'Number of Orders'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model training section
    st.header("Model Training and Prediction")
    
    # Model options
    model_option = st.selectbox(
        "Select Model",
        ["Prophet", "LSTM"]
    )
    
    # Prediction period
    prediction_days = st.slider("Prediction Period (Days)", 1, 14, 7)
    
    # Train model button
    if st.button("Train Model and Predict"):
        with st.spinner("Training model and generating predictions..."):
            # Train Prophet model
            if model_option == "Prophet":
                model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                model.fit(processed_data)
                
                # Create future dataframe for prediction
                future = model.make_future_dataframe(periods=prediction_days * 24 if agg_option == "Hourly" else prediction_days)
                forecast = model.predict(future)
                
                # Save to session state
                st.session_state.model = model
                st.session_state.forecast = forecast
                st.session_state.is_trained = True
                
                st.success("Model trained successfully!")
            
            # Train LSTM model
            elif model_option == "LSTM":
                from preprocessing import prepare_for_lstm
                
                # Prepare data for LSTM
                n_steps = 24  # Number of time steps to use for each sequence
                X, y = prepare_for_lstm(processed_data, n_steps=n_steps)
                
                # Reshape X to be [samples, time steps, features]
                X = X.reshape((X.shape[0], X.shape[1], 1))
                
                # Build LSTM model
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1), return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(50, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                
                # Train model
                model.fit(X, y, epochs=50, verbose=0, batch_size=32)
                
                # Generate predictions
                # First, use the model to predict the next n_steps
                last_sequence = X[-1].reshape(1, n_steps, 1)
                predictions = []
                
                for i in range(prediction_days * 24 if agg_option == "Hourly" else prediction_days):
                    # Predict next value
                    next_pred = model.predict(last_sequence, verbose=0)[0][0]
                    predictions.append(next_pred)
                    
                    # Update sequence for next prediction
                    last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred]]], axis=1)
                
                # Create forecast dataframe similar to Prophet's output
                # Calculate the time delta based on aggregation option
                time_delta = pd.Timedelta(hours=1) if agg_option == "Hourly" else pd.Timedelta(days=1)
                
                # Generate future dates
                future_dates = pd.date_range(
                    start=processed_data['ds'].iloc[-1] + time_delta,
                    periods=len(predictions),
                    freq='H' if agg_option == "Hourly" else 'D'
                )
                
                forecast = pd.DataFrame({
                    'ds': pd.concat([processed_data['ds'], pd.Series(future_dates)]),
                    'yhat': np.append(processed_data['y'].values, predictions),
                    'yhat_lower': np.append(processed_data['y'].values, predictions) * 0.9,  # Simple approximation
                    'yhat_upper': np.append(processed_data['y'].values, predictions) * 1.1   # Simple approximation
                })
                
                # Save to session state
                st.session_state.model = model
                st.session_state.forecast = forecast
                st.session_state.is_trained = True
                st.session_state.model_type = "LSTM"  # Track model type
                
                st.success("LSTM model trained successfully!")
    
    # Display prediction results if available
    if st.session_state.is_trained and st.session_state.forecast is not None:
        st.subheader("Prediction Results")
        
        # Plot forecast
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=processed_data['ds'],
            y=processed_data['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add prediction
        forecast = st.session_state.forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Prediction',
            line=dict(color='red')
        ))
        
        # Add prediction interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Prediction Interval'
        ))
        
        fig.update_layout(
            title=f"Order Prediction for Next {prediction_days} Days",
            xaxis_title="Date",
            yaxis_title="Number of Orders",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast components
        if model_option == "Prophet" and 'model_type' not in st.session_state:
            st.subheader("Forecast Components")
            fig_comp = st.session_state.model.plot_components(st.session_state.forecast)
            st.pyplot(fig_comp)
        
        # Option to save model
        if st.button("Save Model"):
            if 'model_type' in st.session_state and st.session_state.model_type == "LSTM":
                model_filename = 'lstm_model.h5'
                st.session_state.model.save(model_filename)
            else:
                model_filename = 'prophet_model.pkl'
                with open(model_filename, 'wb') as f:
                    pickle.dump(st.session_state.model, f)
            st.success(f"Model saved successfully as {model_filename}!")

else:
    st.info("Please generate or upload data to begin")

# Footer
st.markdown("---")
st.markdown("© 2023 Management System")