import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import joblib # For saving the scaler and encoder

# --- Configuration ---
FILE_PATH = 'dataset/crop_recommendation.csv'
MODEL_NAME = 'recommendation_model.h5'
PREPROCESSOR_NAME = 'recommendation_preprocessor.joblib'
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 32

# --- 1. Data Loading and Preprocessing ---
print(f"Loading data from {FILE_PATH}...")
data = pd.read_csv(FILE_PATH)

# Features (X) and Target (y)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_labels = data['label']

# 1. Scaling Numerical Features (X)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 2. Encoding Categorical Target (y)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_labels)
# Convert encoded labels to one-hot for Keras
num_classes = len(encoder.classes_)
y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_one_hot, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
)

# --- 2. Model Definition (Deep Neural Network) ---
print("Building and compiling the Classification Model...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax') # Output layer: softmax for multi-class
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3. Model Training ---
print(f"Training started for {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1 # Use 0 for silent training
)

# --- 4. Model Evaluation and Saving ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# Save model and preprocessors for Flask deployment
model.save(f'models/{MODEL_NAME}')
joblib.dump(scaler, f'models/{PREPROCESSOR_NAME}_scaler.joblib')
joblib.dump(encoder, f'models/{PREPROCESSOR_NAME}_encoder.joblib')
print(f"Model and preprocessors saved to the 'models/' directory.")

# --- 5. Accuracy and Loss Graphs ---
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Save the figure
plt.tight_layout()
plt.savefig('static/recommendation_accuracy_loss.png')
plt.show()

#