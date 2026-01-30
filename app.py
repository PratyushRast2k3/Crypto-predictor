import os
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime as dt
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(BASE_DIR,"crypto_model.h5")
SCALER_PATH=os.path.join(BASE_DIR,"scaler.pkl")

### G4JYZQVWW3UWV6D9.api key alpha vantage

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    crypto = request.args.get('crypto', 'BTC')  # Default to BTC
    currency = request.args.get('currency', 'USD')  # Default to USD
    days = int(request.args.get('days', 1))  # Default to 1 day prediction

    try:
        # Fetch data
        start = dt.datetime(2016, 1, 1)
        end = dt.datetime.now()
        data = yf.download(f"{crypto}-{currency}", start=start, end=end)

        if data.empty:
            return jsonify({'error': 'Failed to fetch data. Check the ticker symbol.'}), 400

        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Prepare data for prediction
        prediction_days = 60  # The number of days the model is trained for
        model_inputs = scaled_data[-prediction_days:]
        model_inputs = np.array(model_inputs).reshape(1, model_inputs.shape[0], 1)

        # Predict the price for the selected day
        for _ in range(days):
            prediction = model.predict(model_inputs)
            predicted_price = scaler.inverse_transform(prediction)

            # Update model inputs with the latest prediction (to predict the next day)
            model_inputs = np.append(model_inputs[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

        # Convert predicted_price from numpy.float32 to Python float using .item()
        predicted_price_float = predicted_price[0][0].item()

        # Return only the predicted price for the selected day
        return jsonify({
            'predicted_price': predicted_price_float,
            'prediction_day': days
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
