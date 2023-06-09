import os
import sys
from prophet import Prophet
import pandas as pd


def save_data(d, f, columns):
    d[columns].to_csv(f, index=False)


horizon = int(sys.argv[1])
dates = str(sys.argv[2]).split(",")
sales = [int(s) for s in str(sys.argv[3]).split(",")]
frequency = "D"

df = pd.DataFrame({"ds": dates, "y": sales})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=horizon, freq=frequency, include_history=True)
forecast = model.predict(future)

forecast.rename(columns={'yhat': 'y'}, inplace=True)
save_data(forecast.tail(horizon), "C:\\Users\\User\\Desktop\\forecast_day.csv", ["ds", "y"])

if os.path.exists("C:\\Users\\User\\Desktop\\forecast_day.csv"):
    print("SUCCESS")
else:
    print("FAIL")
