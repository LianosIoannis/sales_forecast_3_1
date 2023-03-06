from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go

# Load sales data
df = pd.read_csv("C:\\Users\\User\\Desktop\\sales.csv", sep=",", parse_dates=["ds"], dayfirst=True)

# Create and fit Prophet model
m = Prophet()
m.fit(df)

# Generate future dataframe
future = m.make_future_dataframe(periods=30)

# Make predictions for the future
forecast = m.predict(future)

# Create a Plotly figure with the forecast data
fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Sales'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.2)', name='Upper Bound'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.2)', name='Lower Bound'))

fig.update_layout(title='Sales Forecast',
                  xaxis_title='Date',
                  yaxis_title='Sales')

# Show the figure
fig.show()
