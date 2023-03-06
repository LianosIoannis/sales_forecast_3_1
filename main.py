from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\User\\Desktop\\sales.csv", sep=",", parse_dates=["ds"], dayfirst=True)
print(df.head(10))
#df.plot()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=30)
#future.tail()

forecast = m.predict(future)
print(forecast[['ds', 'yhat']].tail(30))

plt.plot(forecast["ds"], forecast["yhat"])
plt.show()


fig = m.plot(forecast)
plot_plotly(forecast)
plot_components_plotly(forecast)
