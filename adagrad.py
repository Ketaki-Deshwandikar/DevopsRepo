import os
import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, filtfilt, periodogram
import scipy.stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import GRU, Dropout, Dense, Bidirectional, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import mode
import warnings
import random

warnings.filterwarnings("ignore")

# Fix random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define parameters
Channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

T = 150  # Sample Period, Seconds
fs = 128  # Sample rate, Hz
nyq = 0.5 * fs
order = 1
n = int(T * fs) + 1  # Total number of samples to standardize

# Bandpass filter coefficients (3 Hz high-pass and 40 Hz low-pass)
normal_cutoff = [3 / nyq, 40 / nyq]
b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)

train_folder = "C:\\Users\\Ketaki\\Desktop\\Dataset\\training"
test_folder = "C:\\Users\\Ketaki\\Desktop\\Dataset\\test"

# Function to process data from a folder
def process_data(folder_path):
    subject_data = []
    for file in os.listdir(folder_path):
        if file.endswith("_hi.txt"):  # Process only multitasking files
            file_path = os.path.join(folder_path, file)
            Data = pd.read_csv(file_path, sep="  ", header=None, engine="python")
            Data.columns = Channels

            print(f"Processing File: {file}")
            subject_features = pd.DataFrame()

            for Ch in Channels:
                # Pre-Processing: Bandpass Filter
                filtered_signal = filtfilt(b, a, Data[Ch])

                # Apply Discrete Wavelet Transform (DWT)
                coeffs = pywt.wavedec(filtered_signal, wavelet='db4', level=4)

                # Extract Approximation and Details coefficients from DWT
                for i, coef in enumerate(coeffs):
                    coef = np.pad(coef, (0, max(0, n - len(coef))), mode='constant')[:n]
                    subject_features[f"{Ch}_dwt_{i}"] = coef

                # Adding statistical features
                subject_features[f"{Ch}_mean"] = np.mean(filtered_signal)
                subject_features[f"{Ch}_var"] = np.var(filtered_signal)
                subject_features[f"{Ch}_skewness"] = scipy.stats.skew(filtered_signal)
                subject_features[f"{Ch}_kurtosis"] = scipy.stats.kurtosis(filtered_signal)

                # Frequency Analysis to detect Beta Waves (13-30 Hz)
                f, Pxx = periodogram(filtered_signal, fs)
                beta_power = np.sum(Pxx[(f >= 13) & (f <= 30)])  # Power in the beta range
                subject_features[f"{Ch}_beta_power"] = beta_power

            subject_data.append(subject_features)

    return pd.concat(subject_data, ignore_index=True) if subject_data else pd.DataFrame()
def process_file(data):
    subject_features = pd.DataFrame()
    for Ch in Channels:
        # Pre-Processing: Bandpass Filter
        filtered_signal = filtfilt(b, a, data[Ch])
        # Apply Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec(filtered_signal, wavelet="db4", level=4)
        # Extract Approximation and Details coefficients from DWT
        for i, coef in enumerate(coeffs):
            coef = np.pad(coef, (0, max(0, n - len(coef))), mode="constant")[:n]
            subject_features[f"{Ch}dwt{i}"] = coef
        # Adding statistical features
        subject_features[f"{Ch}_mean"] = np.mean(filtered_signal)
        subject_features[f"{Ch}_var"] = np.var(filtered_signal)
        subject_features[f"{Ch}_skewness"] = scipy.stats.skew(filtered_signal)
        subject_features[f"{Ch}_kurtosis"] = scipy.stats.kurtosis(filtered_signal)
        # Frequency Analysis to detect Beta Waves (13-30 Hz)
        f, Pxx = periodogram(filtered_signal, fs)
        beta_power = np.sum(Pxx[(f >= 13) & (f <= 30)])  # Power in the beta range
        subject_features[f"{Ch}_beta_power"] = beta_power
    return subject_features

# Prepare the data
def prepare_data(data):
    beta_powers = []
    for index, row in data.iterrows():
        avg_beta_power = np.mean([row[f"{Ch}_beta_power"] for Ch in Channels])
        beta_powers.append(avg_beta_power)

    # Compute thresholds
    low_threshold = np.percentile(beta_powers, 33)
    high_threshold = np.percentile(beta_powers, 66)

    # Assign stress labels
    labels = []
    for beta in beta_powers:
        if beta < low_threshold:
            labels.append("low")
        elif beta < high_threshold:
            labels.append("medium")
        else:
            labels.append("high")

    data["stress_level"] = labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Features and target
    X = data.drop(columns=["stress_level"]).values
    y = y_encoded
    return X, y

# Prepare single file for prediction
def prepare_single_file(file_path):
    Data = pd.read_csv(file_path, sep="  ", header=None, engine="python")
    Data.columns = Channels
    processed_data = process_file(Data)
    return processed_data

# Predict and display accuracy for a single file
def predict_single_file(file_path):
    # Process and prepare the file
    processed_data = prepare_single_file(file_path)
    X_single, y_true = prepare_data(processed_data)

    # Reshape and standardize
    X_single = X_single.reshape(X_single.shape[0], 1, X_single.shape[1])
    X_single = scaler.transform(X_single.reshape(X_single.shape[0], -1)).reshape(
        X_single.shape
    )
    # Predict
    y_pred_cat = model.predict(X_single)
    y_pred = np.argmax(y_pred_cat, axis=1)
    # Calculate accuracy
    test_accuracy = accuracy_score(y_true, y_pred)
   
   
    # Map numerical labels to stress levels
    label_counts = {0: "low", 1: "medium", 2: "high"}

    # Get the most common prediction using scipy.stats.mode
    most_common_label = mode(y_pred, keepdims=True).mode[
        0
    ]  # Access the mode attribute directly

    # Map the most common label to the stress level
    stress_level = label_counts[most_common_label]

    print(f"Predicted Stress Level for Subject: {stress_level}")
    return stress_level


# Load training data
print("Processing Training Data...")
train_data = process_data(train_folder)
X_train, y_train = prepare_data(train_data)

# Load testing data
print("Processing Testing Data...")
test_data = process_data(test_folder)
X_test, y_test = prepare_data(test_data)

# Reshape data for GRU model
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# Convert labels to categorical
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Build the hybrid model
model = Sequential()

# CNN Layer for feature extraction
model.add(Conv1D(128, 1, activation="relu", padding="valid", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.2))

# BiLSTM Layer
model.add(Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(Dropout(0.4))

# First GRU Layer
model.add(Bidirectional(GRU(64, return_sequences=True)))
model.add(Dropout(0.4))

# Second GRU Layer
model.add(Bidirectional(GRU(32, return_sequences=False)))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(3, activation="softmax"))

# Compile the model
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1,
)

# Save the trained model
model.save("adag.keras")
print("Model saved successfully!")

# Load the saved model
loaded_model = load_model("adag.keras")

# Evaluate the model
y_pred_cat = loaded_model.predict(X_test)
y_pred = np.argmax(y_pred_cat, axis=1)

# Calculate Test Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Print Classification Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

def main():
    while True:
        # Prompt user to input the file path
        print("\nPlease provide the path to your file:")
        file_path = input("Enter the file path: ")

        # Check if the provided path is valid
        if os.path.isfile(file_path):
            print(f"Processing file: {os.path.basename(file_path)}")
            predict_single_file(file_path)
        else:
            print("Error: The provided path is not a valid file. Please try again.")

        # Ask the user if they want to test another file
        print("\nWould you like to test another file? (yes/no)")
        user_response = input().strip().lower()
        if user_response not in ["yes", "y"]:
            print("Exiting the testing loop. Goodbye!")
            break


if __name__ == "__main__":
    main()
# Plot Training and Test Accuracy
plt.figure(figsize=(10, 6))

# Training accuracy from history
plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue", linewidth=2)

# Add a horizontal line for test accuracy
plt.axhline(y=test_accuracy, color="green", linestyle="--", label=f"Test Accuracy: {test_accuracy:.2f}", linewidth=2)

# Adjust Y-axis limits to focus on accuracy range
plt.ylim([0.8, 1.0])

# Add labels, title, and legend
plt.title("Model Accuracy: Training vs Test", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange", linestyle="--", linewidth=2)
plt.title("Model Accuracy Over Epochs", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.8, 1.0])  # Adjust Y-axis to focus on key range
plt.show()

# Plot ROC Curve
plt.figure(figsize=(8, 6))
# #Plot ROC curve for the first class (medium)
fpr, tpr, _ = roc_curve(y_test_cat[:, 1], y_pred_cat[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.6)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("Receiver Operating Characteristic", fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


#Plot ROC curve for the first class (medium)
fpr, tpr, _ = roc_curve(y_test_cat[:, 1], y_pred_cat[:, 1])
roc_auc = auc(fpr, tpr)

#plot trrain and test accuracy
# Evaluate the test accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

# Plot training accuracy and test accuracy
plt.figure(figsize=(10, 6))

# Training accuracy from history
plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue")

# Add a horizontal line for test accuracy
plt.axhline(y=test_accuracy, color="green", linestyle="--", label=f"Test Accuracy: {test_accuracy:.2f}")

# Add labels, title, and legend
plt.title("Model Accuracy: Training vs Test", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)

# Display the plot
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
plt.title("Model Accuracy Over Epochs", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.show()

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()
