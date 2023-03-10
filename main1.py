from prophet import Prophet
import pandas as pd
from pathlib import Path
from prophet.diagnostics import performance_metrics, cross_validation
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt


# sales_filename = Path(sys.argv[1])

sales_filename = Path("C:\\Users\\User\\Desktop\\SALES_FOLDER\\45628_sales.csv")

df = pd.read_csv(sales_filename, sep=",", parse_dates=["ds"], dayfirst=True)

m = Prophet()

m.fit(df)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

sales_filename_forecast = sales_filename.parent / (sales_filename.stem + "_forecast_1.csv")

forecast.to_csv(sales_filename_forecast, index=False, columns=['ds', 'yhat'])

df_cv = cross_validation(m, initial='1800 days', period='180 days', horizon='90 days')

# Compute performance metrics
df_metrics = performance_metrics(df_cv)

# Print performance metrics
print(df_metrics)

plt.switch_backend('agg')
fig = m.plot(forecast)
fig.savefig("C:\\Users\\User\\Desktop\\SALES_FOLDER\\figure")


fig = plot_plotly(m, forecast)
fig.show()

# Plot the trend and components
fig = plot_components_plotly(m, forecast)
fig.show()
