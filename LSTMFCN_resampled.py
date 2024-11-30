import numpy as np
import pandas as pd
from sktime.classification.deep_learning import LSTMFCNClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.optimizers import Adam

print("Results LSTMFCN_resampled with attention.py")

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

# Resample MITBIH to balance classes
df_mitbih_train = pd.DataFrame(X_train_mitbih)
df_mitbih_train['label'] = y_train_mitbih

# Resample each class to 20,000 instances
df_resampled = pd.concat([
    resample(df_mitbih_train[df_mitbih_train['label'] == i], 
             replace=True, n_samples=20000, random_state=i) for i in range(5)
])

# Extract features and labels after resampling
X_train_mitbih = df_resampled.iloc[:, :-1].values
y_train_mitbih = df_resampled['label'].values
X_train_mitbih_3D = X_train_mitbih[:, np.newaxis, :]
X_test_mitbih_3D = X_test_mitbih[:, np.newaxis, :]

# Define and train model for MITBIH
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

# Process PTBDB Dataset
ptbdb_abnormal.columns = range(ptbdb_abnormal.shape[1])
ptbdb_normal.columns = range(ptbdb_normal.shape[1])

# Combine PTBDB datasets
ptbdb_combined = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True)
X_ptbdb = ptbdb_combined.iloc[:, :-1].values
y_ptbdb = ptbdb_combined.iloc[:, -1].values

# Split and resample PTBDB
X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(
    X_ptbdb, y_ptbdb, test_size=0.2, random_state=42, stratify=y_ptbdb
)

# Resample classes in PTBDB to balance
df_ptbdb_train = pd.DataFrame(X_train_ptbdb)
df_ptbdb_train['label'] = y_train_ptbdb
df_normal = df_ptbdb_train[df_ptbdb_train['label'] == 0]
df_abnormal = df_ptbdb_train[df_ptbdb_train['label'] == 1]
sample_size = max(len(df_normal), len(df_abnormal))

df_normal_resampled = resample(df_normal, replace=True, n_samples=sample_size, random_state=42)
df_abnormal_resampled = resample(df_abnormal, replace=True, n_samples=sample_size, random_state=123)
df_ptbdb_resampled = pd.concat([df_normal_resampled, df_abnormal_resampled])

# Extract features and labels after resampling
X_train_ptbdb = df_ptbdb_resampled.iloc[:, :-1].values
y_train_ptbdb = df_ptbdb_resampled['label'].values
X_train_ptbdb_3D = X_train_ptbdb[:, np.newaxis, :]
X_test_ptbdb_3D = X_test_ptbdb[:, np.newaxis, :]

# Define and train model for PTBDB
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
f1_ptbdb = f1_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
recall_ptbdb = recall_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
precision_ptbdb = precision_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
conf_matrix_ptbdb = confusion_matrix(y_test_ptbdb, predictions_ptbdb)

# Display results
print(f"\nPTBDB Dataset Results:")
print(f"  - Accuracy: {accuracy_ptbdb:.4f}")
print(f"  - F1 Score: {f1_ptbdb:.4f}")
print(f"  - Recall: {recall_ptbdb:.4f}")
print(f"  - Precision: {precision_ptbdb:.4f}")
print("  - Confusion Matrix:\n", conf_matrix_ptbdb)

# Save results summary to CSV
results = {
    "Dataset": ["MITBIH", "PTBDB"],
    "Accuracy": [accuracy_mitbih, accuracy_ptbdb],
    "F1 Score": [f1_mitbih, f1_ptbdb],
    "Recall": [recall_mitbih, recall_ptbdb],
    "Precision": [precision_mitbih, precision_ptbdb]
}
results_df = pd.DataFrame(results)
results_df.to_csv("LSTMFCN_resampled_results.csv", index=False)

# Save predictions with true labels to CSV
mitbih_predictions_df = pd.DataFrame({"True Label": y_test_mitbih, "Predicted Label": predictions_mitbih})
ptbdb_predictions_df = pd.DataFrame({"True Label": y_test_ptbdb, "Predicted Label": predictions_ptbdb})

mitbih_predictions_df.to_csv("mitbih_predictions_LSTMFCN_resampled.csv", index=False)
ptbdb_predictions_df.to_csv("ptbdb_predictions_LSTMFCN_resampled.csv", index=False)

print("\nResampled results and predictions saved to CSV files.")
