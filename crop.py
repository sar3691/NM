import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

try:
    # Load dataset
    file_path = "/mnt/data/Crop_recommendation (1).csv"
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit()

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Exploratory Data Analysis
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

# Selecting features and target
target_column = "label"
try:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical features to numerical if any exist
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"Converting categorical columns to numerical: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Check if we have enough data
    if len(X) < 10:
        raise ValueError("Dataset is too small for meaningful training.")

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance Score")
    plt.show()

    # Save model
    joblib.dump(model, "crop_prediction_model.pkl")
    print("Model successfully saved as 'crop_prediction_model.pkl'.")

    # Prediction function for new data
    def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
        """
        Predict crop type based on input features
        Parameters: N (Nitrogen), P (Phosphorus), K (Potassium), temperature (°C), humidity (%), ph (soil pH), rainfall (mm)
        Returns: Predicted crop
        """
        new_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        prediction = model.predict(new_data)[0]
        return label_encoder.inverse_transform([prediction])[0]

    # Example Predictions
    print("\nExample Predictions:")
    examples = [
        (90, 42, 43, 20.5, 80, 6.5, 200),
        (30, 20, 10, 30.0, 60, 5.5, 150),
        (100, 50, 50, 25.0, 70, 6.8, 250)
    ]
    
    for n, p, k, temp, hum, ph_val, rain in examples:
        result = predict_crop(n, p, k, temp, hum, ph_val, rain)
        print(f"\nN: {n}, P: {p}, K: {k}, Temp: {temp}°C, Humidity: {hum}%, pH: {ph_val}, Rainfall: {rain}mm")
        print(f"Predicted Crop: {result}")

except ValueError as e:
    print(f"Error: {str(e)}")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
