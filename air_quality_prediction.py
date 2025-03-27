import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
air_quality_data = pd.read_csv("co2_emissions_data.csv")

# Ensure column names are stripped of extra spaces
air_quality_data.columns = air_quality_data.columns.str.strip()

# Define features and target
features = ['Engine Size (L)']  # Input variable
target = 'CO2 Emissions (g/km)'  # Output variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(air_quality_data[features], air_quality_data[target], test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Error evaluation
print("CO2 Emissions Prediction - Mean Absolute Error:", mean_absolute_error(y_test, predictions))
