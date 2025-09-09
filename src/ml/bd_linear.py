from adapters import csv_adapter as adapter
import pandas as pd
import prophet as pf

# Load the dataset
df = pd.read_csv("path_to_your_file.csv")

# Initialize the Prophet model
model = pf.Prophet()

# Fit the model on the dataset
model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Save the forecast to a CSV file
forecast.to_csv("forecasted_data.csv", index=False)
