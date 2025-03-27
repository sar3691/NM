import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
solar_data = pd.read_csv("solar_power_classification.csv")

# Print unique values in Power_Output to understand its type
print("Unique Power Output values:", solar_data["Power_Output"].unique())

# Convert categorical Power_Output values to numerical labels
power_mapping = {"Low": 0, "Medium": 1, "High": 2}  # Adjust if needed
solar_data["Power_Output"] = solar_data["Power_Output"].map(power_mapping)

# Define threshold for maintenance (e.g., "Low" power output requires maintenance)
solar_data["Maintenance_Needed"] = (solar_data["Power_Output"] == 0).astype(int)

# Separate features and target variable
X = solar_data.drop(columns=["Power_Output", "Maintenance_Needed"])
y = solar_data["Maintenance_Needed"]

# Convert categorical features (if any) to numerical
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
print("Solar Maintenance Prediction - Accuracy:", accuracy)
