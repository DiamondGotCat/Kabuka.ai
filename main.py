# 0. Import Libraries
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd

# 1. Set information and Get Data
start = datetime.datetime(2009, 1, 1)
end = datetime.datetime(2024, 7, 29)
df = yf.download("AAPL", start=start, end=end)

# Prepare data for Prophet
df.reset_index(inplace=True)
df = df[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# 2. Initialize and fit the model
model = Prophet()
model.fit(df)

# 3. Make future dataframe and predict
future = model.make_future_dataframe(periods=1460)  # Forecasting for 365 days into the future
forecast = model.predict(future)

# 4. Visualize the forecast
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['ds'], df['y'], label='Historical Data', color='blue')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)
ax.set_title('AAPL Stock Price Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.legend()
plt.grid(True)
plt.show()
