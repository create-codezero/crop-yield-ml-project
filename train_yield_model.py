import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import joblib # For saving the preprocessor
from scipy.sparse import issparse

# --- Configuration ---
FILE_PATH = 'dataset/merged_crop_yield_data.csv' # ASSUME THIS FILE EXISTS (MERGED DATA)
MODEL_NAME = 'yield_prediction_model.h5'
PREPROCESSOR_NAME = 'yield_preprocessor.joblib'
RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 64

# --- 1. Data Loading and Preprocessing ---
print(f"Loading merged data from {FILE_PATH}...")
# Assuming you have merged the files and calculated Yield
# NOTE: The provided code assumes this file exists.
# data = pd.read_csv(FILE_PATH) 
# For demonstration purposes, I will create dummy data to ensure the script runs.
try:
    data = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"File not found at {FILE_PATH}. Creating dummy data for demonstration.")
    np.random.seed(RANDOM_SEED)
    N = 1000
    data = pd.DataFrame({
        'State_Name': np.random.choice(['StateA', 'StateB', 'StateC'], N),
        'District_Name': np.random.choice(['Dist1', 'Dist2'], N),
        'Season': np.random.choice(['Kharif', 'Rabi'], N),
        'Crop': np.random.choice(['Rice', 'Wheat', 'Maize'], N),
        'Area': np.random.uniform(10, 1000, N),
        'Production': np.random.uniform(500, 50000, N),
        'N': np.random.uniform(50, 150, N),
        'P': np.random.uniform(20, 80, N),
        'K': np.random.uniform(10, 50, N),
        'temperature': np.random.uniform(15, 35, N),
        'humidity': np.random.uniform(40, 90, N),
        'ph': np.random.uniform(5.5, 7.5, N),
        'rainfall': np.random.uniform(500, 2500, N),
    })


# 1. Feature Engineering: Calculate Yield (Target Variable)
data['Yield_kg_per_ha'] = data['Production'] / data['Area']
# Handle potential infinite/NaN values from Area=0 or missing data
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['Yield_kg_per_ha', 'Area', 'Production'], inplace=True)
data = data[(data['Yield_kg_per_ha'] > 0)] # Filter out records with zero or negative yield

# Target (y)
y = data['Yield_kg_per_ha'].values

# Features (X): Include both numerical and categorical features
numerical_features = ['Area', 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
categorical_features = ['State_Name', 'District_Name', 'Season', 'Crop']
X = data[numerical_features + categorical_features]

# 2. Preprocessing Pipeline: Scaling and One-Hot Encoding
# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features), # Scale numerical features
        # --- FIX APPLIED HERE: sparse_output=False ---
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # One-Hot Encode categories and ensure dense output
    ],
    remainder='passthrough'
)

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Optional: Check if the output is dense (it should be now)
if issparse(X_processed):
    print("Warning: X_processed is still sparse. Converting to dense array.")
    X_processed = X_processed.toarray()
else:
    print("X_processed is a dense NumPy array (Correct for Keras).")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=RANDOM_SEED
)

# --- 2. Model Definition (Deep Neural Network Regressor) ---
print("Building and compiling the Regression Model...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear') # Output layer: linear for regression
])

model.compile(
    optimizer='adam',
    loss='mae', # Mean Absolute Error is common for regression loss
    metrics=['mse', 'mae'] # Mean Squared Error (MSE) and MAE for monitoring
)

# --- 3. Model Training ---
print(f"Training started for {EPOCHS} epochs...")
# The model.fit() will now work correctly with validation_split
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# --- 4. Model Evaluation and Saving ---
loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE (Mean Absolute Error): {mae:.2f} kg/ha")

# Create a directory for models if it doesn't exist
import os
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True) # Assuming 'static' is for graphs

# Save model and preprocessor for Flask deployment
model.save(f'models/{MODEL_NAME}')
joblib.dump(preprocessor, f'models/{PREPROCESSOR_NAME}')
print(f"Model and preprocessor saved to the 'models/' directory.")

# --- 5. Loss Graphs ---
plt.figure(figsize=(12, 5))

# Loss Plot (MAE)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Loss (Mean Absolute Error)')
plt.ylabel('MAE (kg/ha)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Save the figure
plt.tight_layout()
plt.savefig('static/yield_loss_mae.png')
plt.show()