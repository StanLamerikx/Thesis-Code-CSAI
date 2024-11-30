import numpy as np
import pandas as pd
from sktime.classification.deep_learning import LSTMFCNClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

print("Results LSTMFCN with attention.py")

# Load datasets
ptbdb_abnormal = pd.read_csv('./ptbdb_abnormal.csv')
ptbdb_normal = pd.read_csv('./ptbdb_normal.csv')
mitbih_test = pd.read_csv('./mitbih_test.csv')
mitbih_train = pd.read_csv('./mitbih_train.csv')

# Process MITBIH Dataset
y_train_mitbih = mitbih_train.iloc[:, -1].values
y_test_mitbih = mitbih_test.iloc[:, -1].values
X_train_mitbih = mitbih_train.iloc[:, :-1].values
X_test_mitbih = mitbih_test.iloc[:, :-1].values

# Convert MITBIH data to numpy 3D format (n_instances, 1, 187)
X_train_mitbih_3D = X_train_mitbih[:, np.newaxis, :]
X_test_mitbih_3D = X_test_mitbih[:, np.newaxis, :]

# Train model for MITBIH
# Define the model classifier with recommended hyperparameters
model_mitbih = LSTMFCNClassifier(
    n_epochs=100,               # Set to a suitable number of epochs
    batch_size=32,               # Batch size for training
    random_state=42,             # Seed for reproducibility
    attention=True
)

model_mitbih.fit(X_train_mitbih_3D, y_train_mitbih)

# Predictions and evaluation for MITBIH
predictions_mitbih = model_mitbih.predict(X_test_mitbih_3D)
accuracy_mitbih = accuracy_score(y_test_mitbih, predictions_mitbih)
f1_mitbih = f1_score(y_test_mitbih, predictions_mitbih, average='weighted')
recall_mitbih = recall_score(y_test_mitbih, predictions_mitbih, average='weighted')
precision_mitbih = precision_score(y_test_mitbih, predictions_mitbih, average='weighted')
conf_matrix_mitbih = confusion_matrix(y_test_mitbih, predictions_mitbih)

# Print shape of MITBIH outputs
print("Shape of y_test_mitbih:", y_test_mitbih.shape)
print("Shape of predictions_mitbih:", predictions_mitbih.shape)

print(f"\nMITBIH Dataset Results:")
print(f"  - Accuracy: {accuracy_mitbih:.4f}")
print(f"  - F1 Score: {f1_mitbih:.4f}")
print(f"  - Recall: {recall_mitbih:.4f}")
print(f"  - Precision: {precision_mitbih:.4f}")
print("  - Confusion Matrix:\n", conf_matrix_mitbih)

# Rename columns to integers
ptbdb_abnormal.columns = range(ptbdb_abnormal.shape[1])
ptbdb_normal.columns = range(ptbdb_normal.shape[1])

# Combine the datasets
ptbdb_combined = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True)

# Separate features and labels
X_ptbdb = ptbdb_combined.iloc[:, :-1].values  # All columns except the last one are features
y_ptbdb = ptbdb_combined.iloc[:, -1].values   # Last column is the label

# Split into train and test sets
X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(
    X_ptbdb, y_ptbdb, test_size=0.2, random_state=42, stratify=y_ptbdb
)

# Reshape data to fit format (n_instances, 1, n_features)
X_train_ptbdb_3D = X_train_ptbdb[:, np.newaxis, :]
X_test_ptbdb_3D = X_test_ptbdb[:, np.newaxis, :]

# Train model on PTBDB
# Define the model classifier with recommended hyperparameters
model_ptbdb = LSTMFCNClassifier(
    n_epochs=100,               # Set to a suitable number of epochs
    batch_size=32,               # Batch size for training
    random_state=42,             # Seed for reproducibility
    attention=False
)

model_ptbdb.fit(X_train_ptbdb_3D, y_train_ptbdb)

# Predictions and evaluation for PTBDB
predictions_ptbdb = model_ptbdb.predict(X_test_ptbdb_3D)
accuracy_ptbdb = accuracy_score(y_test_ptbdb, predictions_ptbdb)
f1_ptbdb = f1_score(y_test_ptbdb, predictions_ptbdb)
recall_ptbdb = recall_score(y_test_ptbdb, predictions_ptbdb)
precision_ptbdb = precision_score(y_test_ptbdb, predictions_ptbdb)
conf_matrix_ptbdb = confusion_matrix(y_test_ptbdb, predictions_ptbdb)

# Display results
print(f"\nPTBDB Dataset Results:")
print(f"  - Accuracy: {accuracy_ptbdb:.4f}")
print(f"  - F1 Score: {f1_ptbdb:.4f}")
print(f"  - Recall: {recall_ptbdb:.4f}")
print(f"  - Precision: {precision_ptbdb:.4f}")
print("  - Confusion Matrix:\n", conf_matrix_ptbdb)

# Define results dictionary after obtaining metrics for both datasets
results = {
    "Dataset": ["MITBIH", "PTBDB"],
    "Accuracy": [accuracy_mitbih, accuracy_ptbdb],
    "F1 Score": [f1_mitbih, f1_ptbdb],
    "Recall": [recall_mitbih, recall_ptbdb],
    "Precision": [precision_mitbih, precision_ptbdb]
}

# Save results summary to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("LSTMFCN_results.csv", index=False)

# Save predictions with true labels to CSV
mitbih_predictions_df = pd.DataFrame({"True Label": y_test_mitbih, "Predicted Label": predictions_mitbih})
ptbdb_predictions_df = pd.DataFrame({"True Label": y_test_ptbdb, "Predicted Label": predictions_ptbdb})

mitbih_predictions_df.to_csv("mitbih_predictions_LSTMFCN.csv", index=False)
ptbdb_predictions_df.to_csv("ptbdb_predictions_LSTMFCN.csv", index=False)

print("\nResults and predictions saved to CSV files.")
