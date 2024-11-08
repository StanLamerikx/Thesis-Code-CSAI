import numpy as np
import pandas as pd
from sktime.classification.kernel_based import RocketClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

print("Results RocketClassifier_resampled.py")

# Load datasets
ptbdb_abnormal = pd.read_csv('./ptbdb_abnormal.csv')
ptbdb_normal = pd.read_csv('./ptbdb_normal.csv')
mitbih_test = pd.read_csv('./mitbih_test.csv')
mitbih_train = pd.read_csv('./mitbih_train.csv')

# 1. MITBIH Dataset (resampling for balance)
y_train_mitbih = mitbih_train.iloc[:, -1].values
y_test_mitbih = mitbih_test.iloc[:, -1].values
X_train_mitbih = mitbih_train.iloc[:, :-1].values
X_test_mitbih = mitbih_test.iloc[:, :-1].values

# Create a DataFrame for resampling
df_mitbih_train = pd.DataFrame(X_train_mitbih)
df_mitbih_train['label'] = y_train_mitbih

# Separate and resample classes for MITBIH
df_0 = df_mitbih_train[df_mitbih_train['label'] == 0].sample(n=20000, random_state=42)
df_1 = resample(df_mitbih_train[df_mitbih_train['label'] == 1], replace=True, n_samples=20000, random_state=123)
df_2 = resample(df_mitbih_train[df_mitbih_train['label'] == 2], replace=True, n_samples=20000, random_state=124)
df_3 = resample(df_mitbih_train[df_mitbih_train['label'] == 3], replace=True, n_samples=20000, random_state=125)
df_4 = resample(df_mitbih_train[df_mitbih_train['label'] == 4], replace=True, n_samples=20000, random_state=126)

df_mitbih_resampled = pd.concat([df_0, df_1, df_2, df_3, df_4])

# Extract features and labels after resampling
X_train_mitbih = df_mitbih_resampled.iloc[:, :-1].values
y_train_mitbih = df_mitbih_resampled['label'].values
X_train_mitbih_3D = X_train_mitbih[:, np.newaxis, :]
X_test_mitbih_3D = X_test_mitbih[:, np.newaxis, :]

# Train RocketClassifier for MITBIH
rocket_classifier_mitbih = RocketClassifier(num_kernels=10000, n_jobs=-1, random_state=42)
rocket_classifier_mitbih.fit(X_train_mitbih_3D, y_train_mitbih)

# Predictions and evaluation for MITBIH
predictions_mitbih = rocket_classifier_mitbih.predict(X_test_mitbih_3D)
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

# Rename columns to integers for PTBDB
ptbdb_abnormal.columns = range(ptbdb_abnormal.shape[1])
ptbdb_normal.columns = range(ptbdb_normal.shape[1])

# Combine PTBDB datasets
ptbdb_combined = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True)

# Split combined PTBDB into features and labels
X_ptbdb = ptbdb_combined.iloc[:, :-1].values
y_ptbdb = ptbdb_combined.iloc[:, -1].values

X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(
    X_ptbdb, y_ptbdb, test_size=0.2, random_state=42, stratify=y_ptbdb
)

# Create a DataFrame for resampling the PTBDB training set
df_ptbdb_train = pd.DataFrame(X_train_ptbdb)
df_ptbdb_train['label'] = y_train_ptbdb

# Separate and resample classes for PTBDB to balance classes in the training set
df_normal = df_ptbdb_train[df_ptbdb_train['label'] == 0]
df_abnormal = df_ptbdb_train[df_ptbdb_train['label'] == 1]
sample_size = max(len(df_normal), len(df_abnormal))

# Resample to balance the classes
df_normal_resampled = resample(df_normal, replace=True, n_samples=sample_size, random_state=42)
df_abnormal_resampled = resample(df_abnormal, replace=True, n_samples=sample_size, random_state=123)
df_ptbdb_resampled = pd.concat([df_normal_resampled, df_abnormal_resampled])

# Extract features and labels after resampling
X_train_ptbdb = df_ptbdb_resampled.iloc[:, :-1].values
y_train_ptbdb = df_ptbdb_resampled['label'].values
X_train_ptbdb_3D = X_train_ptbdb[:, np.newaxis, :]
X_test_ptbdb_3D = X_test_ptbdb[:, np.newaxis, :]

# Train RocketClassifier for PTBDB
rocket_classifier_ptbdb = RocketClassifier(num_kernels=10000, n_jobs=-1, random_state=42)
rocket_classifier_ptbdb.fit(X_train_ptbdb_3D, y_train_ptbdb)

# Predictions and evaluation for PTBDB
predictions_ptbdb = rocket_classifier_ptbdb.predict(X_test_ptbdb_3D)
accuracy_ptbdb = accuracy_score(y_test_ptbdb, predictions_ptbdb)
f1_ptbdb = f1_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
recall_ptbdb = recall_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
precision_ptbdb = precision_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
conf_matrix_ptbdb = confusion_matrix(y_test_ptbdb, predictions_ptbdb)

# Print shape of PTBDB outputs
print("Shape of y_test_ptbdb:", y_test_ptbdb.shape)
print("Shape of predictions_ptbdb:", predictions_ptbdb.shape)

print(f"\nPTBDB Dataset Results:")
print(f"  - Accuracy: {accuracy_ptbdb:.4f}")
print(f"  - F1 Score: {f1_ptbdb:.4f}")
print(f"  - Recall: {recall_ptbdb:.4f}")
print(f"  - Precision: {precision_ptbdb:.4f}")
print("  - Confusion Matrix:\n", conf_matrix_ptbdb)

# Save results and predictions to CSV
results = {
    "Dataset": ["MITBIH", "PTBDB"],
    "Accuracy": [accuracy_mitbih, accuracy_ptbdb],
    "F1 Score": [f1_mitbih, f1_ptbdb],
    "Recall": [recall_mitbih, recall_ptbdb],
    "Precision": [precision_mitbih, precision_ptbdb]
}

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
results_df.to_csv("RocketClassifier_resampled_results.csv", index=False)

# Save predictions with true labels to CSV
mitbih_predictions_df = pd.DataFrame({"True Label": y_test_mitbih, "Predicted Label": predictions_mitbih})
ptbdb_predictions_df = pd.DataFrame({"True Label": y_test_ptbdb, "Predicted Label": predictions_ptbdb})

mitbih_predictions_df.to_csv("mitbih_predictions.csv", index=False)
ptbdb_predictions_df.to_csv("ptbdb_predictions.csv", index=False)

print("\nResults and predictions saved to CSV files.")
