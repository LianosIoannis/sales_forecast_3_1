# import required libraries
import pandas as pd
from neuralprophet import NeuralProphet

# load the data from the CSV file
df = pd.read_csv("C:\\Users\\User\\Desktop\\sales_day.csv", parse_dates=["ds"])

# convert the "ds" column to datetime format

# define the NeuralProphet model
model = NeuralProphet(
    learning_rate=0.01,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode="multiplicative",
    normalize="auto",
    num_hidden_layers=3,
    d_hidden=32,
    trend_reg=0.1,
    seasonality_reg=0.1,
    ar_reg=0.1,
    loss_func="huber"
)

# fit the model to the sales data
metrics = model.fit(df, freq="D", epochs=1000)

# make predictions for the next 30 days
future = model.make_future_dataframe(df, periods=30)
forecast = model.predict(future)

# print the forecasted values for the next 30 days
print(forecast.tail(30))
