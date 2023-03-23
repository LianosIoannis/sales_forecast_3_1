import sys
from prophet import Prophet
import pandas as pd


def print_data(d, n):
    pd.set_option('display.max_rows', None)
    print(d.tail(n))


def save_data(d, f):
    d.to_csv(f, index=False)


#filename = "C:\\Users\\User\\Desktop\\sales_day.csv"
folder = "C:\\Users\\User\\Desktop\\"

#horizon = int(sys.argv[1])
#sales_file = folder + str((sys.argv[2]))
#forecast_file = folder + str((sys.argv[3]))
#frequency = str((sys.argv[4]))

horizon = 15
sales_file = folder + "sales_day.csv"
forecast_file = folder + "forecast_day.csv"
frequency = "D"

data = pd.read_csv(sales_file, parse_dates=["ds"])

model = Prophet()
model.fit(data)

future = model.make_future_dataframe(periods=horizon, freq=frequency, include_history=True)
forecast = model.predict(future)

forecast.rename(columns={'yhat': 'y'}, inplace=True)

print_data(forecast[["ds", "y"]], horizon)
save_data(forecast[["ds", "y"]].tail(horizon), forecast_file)
