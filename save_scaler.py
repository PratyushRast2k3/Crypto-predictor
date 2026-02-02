import joblib
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

# Define the cryptocurrency and currency pair
crypto = 'BTC'
currency = 'USD'

# Fetch data
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()
data = requests.download(f"{crypto}-{currency}", start=start, end=end)

# Ensure the data has the 'Close' column
if 'Close' not in data.columns:
    raise ValueError("Data does not contain 'Close' column.")

# Prepare the data for scaling
training_data = data['Close'].values.reshape(-1, 1)

# Initialize and fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(training_data)

# Save the scaler to a file
joblib.dump(scaler, r"C:\Projects 4th year\Crypto predictor\scaler.pkl")

print("Scaler saved successfully.")
