import sys
from prophet import Prophet
import pandas as pd
from pathlib import Path

sales_filename = Path(sys.argv[1])


df = pd.read_csv(sales_filename, sep=",", parse_dates=["ds"], dayfirst=True)

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

sales_filename_forecast = sales_filename.parent / (sales_filename.stem + "_forecast.csv")

forecast.to_csv(sales_filename_forecast, index=False, columns=['ds', 'yhat'])



