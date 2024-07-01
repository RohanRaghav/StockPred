#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam


# In[2]:


try:
    df = pd.read_csv('portfolio_data.csv')
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
except FileNotFoundError:
    print("File not found. Please check the file path.")
else:
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Display the first few rows to ensure data is read correctly
    print(df.head())

    # List of features to plot
    features = ['AMZN', 'DPZ', 'BTC', 'NFLX']

    # Loop through each feature and create a separate plot
    for feature in features:
        plt.figure(figsize=(14, 7))
        
        # Using seaborn to plot the feature
        sns.lineplot(data=df, x='Date', y=feature, label=feature, marker='o')
        
        # Adding titles and labels
        plt.title(f'{feature} Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        
        # Add a legend
        plt.legend()
        
        # Show the plot
        plt.show()


# In[3]:


try:
    df = pd.read_csv('portfolio_data.csv')
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
except FileNotFoundError:
    print("File not found. Please check the file path.")
else:
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract the year from the Date column
    df['Year'] = df['Date'].dt.year

    # Display the first few rows to ensure data is read correctly
    print(df.head())

    # List of features to plot
    features = ['AMZN', 'DPZ', 'BTC', 'NFLX']

    # Loop through each feature and create separate plots for each year
    for feature in features:
        unique_years = df['Year'].unique()
        
        for year in unique_years:
            plt.figure(figsize=(14, 7))
            
            # Filter data for the specific year
            df_year = df[df['Year'] == year]
            
            # Using seaborn to plot the feature for the specific year
            sns.lineplot(data=df_year, x='Date', y=feature, label=f'{feature} ({year})', marker='o')
            
            # Adding titles and labels
            plt.title(f'{feature} Over Time for {year}', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Value', fontsize=14)
            
            # Rotate date labels for better readability
            plt.xticks(rotation=45)
            
            # Add a legend
            plt.legend()
            
            # Show the plot
            plt.show()


# In[5]:


import numpy as np
# Read the CSV file
try:
    df = pd.read_csv('portfolio_data.csv')
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
except FileNotFoundError:
    print("File not found. Please check the file path.")
else:
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract the year from the Date column
    df['Year'] = df['Date'].dt.year

    # List of features to analyze
    features = ['AMZN', 'DPZ', 'BTC', 'NFLX']

    # Plot original data for reference
    for feature in features:
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=df, x='Date', y=feature, label=feature)
        plt.title(f'{feature} Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    # Preprocess data for each feature
    for feature in features:
        data = df[['Date', feature]].copy()
        data.set_index('Date', inplace=True)

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Prepare the data for the GRU model
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i:i+seq_length]
                y = data[i+seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        SEQ_LENGTH = 30  # Sequence length (e.g., 30 days)
        X, y = create_sequences(scaled_data, SEQ_LENGTH)

        # Split into train and test sets
        SPLIT = int(0.8 * len(X))
        X_train, X_test = X[:SPLIT], X[SPLIT:]
        y_train, y_test = y[:SPLIT], y[SPLIT:]

        # Build the GRU model
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
            GRU(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Plot training history
        plt.figure(figsize=(14, 7))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss for {feature}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.show()

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform the predictions and true values
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred)

        # Plot predictions vs true values
        plt.figure(figsize=(14, 7))
        plt.plot(data.index[-len(y_test):], y_test_inv, label='True Values')
        plt.plot(data.index[-len(y_test):], y_pred_inv, label='Predictions')
        plt.title(f'True vs Predicted Values for {feature}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# Read the CSV file
try:
    df = pd.read_csv('portfolio_data.csv')
except pd.errors.ParserError as e:
    print(f"Error reading CSV file: {e}")
except FileNotFoundError:
    print("File not found. Please check the file path.")
else:
    # Ensure the Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract the year from the Date column
    df['Year'] = df['Date'].dt.year

    # List of features to analyze
    features = ['AMZN', 'DPZ', 'BTC', 'NFLX']

    # Plot original data for reference
    for feature in features:
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=df, x='Date', y=feature, label=feature)
        plt.title(f'{feature} Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    # Function to create sequences for GRU
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    SEQ_LENGTH = 30  # Sequence length (e.g., 30 days)

    # Preprocess data and train the GRU model for each feature
    for feature in features:
        data = df[['Date', feature]].copy()
        data.set_index('Date', inplace=True)

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Create sequences
        X, y = create_sequences(scaled_data, SEQ_LENGTH)

        # Split into train and test sets
        SPLIT = int(0.8 * len(X))
        X_train, X_test = X[:SPLIT], X[SPLIT:]
        y_train, y_test = y[:SPLIT], y[SPLIT:]

        # Build the GRU model
        model = Sequential([
            GRU(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
            GRU(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mean_squared_error')

        # Train the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Plot training history
        plt.figure(figsize=(14, 7))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss for {feature}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.show()

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform the predictions and true values
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred)

        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)

        print(f'{feature} - Mean Squared Error (MSE): {mse}')
        print(f'{feature} - Mean Absolute Error (MAE): {mae}')
        print(f'{feature} - Root Mean Squared Error (RMSE): {rmse}')

        # Plot predictions vs true values
        plt.figure(figsize=(14, 7))
        plt.plot(data.index[-len(y_test):], y_test_inv, label='True Values')
        plt.plot(data.index[-len(y_test):], y_pred_inv, label='Predictions')
        plt.title(f'True vs Predicted Values for {feature}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

