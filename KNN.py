import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load datasets
ptbdb_abnormal = pd.read_csv('./ptbdb_abnormal.csv')
ptbdb_normal = pd.read_csv('./ptbdb_normal.csv')
mitbih_test = pd.read_csv('./mitbih_test.csv')
mitbih_train = pd.read_csv('./mitbih_train.csv')

# Function to extract features and reshape data
def extract_features(data):
    features = pd.DataFrame({
        'mean': data.mean(axis=1),
        'std': data.std(axis=1),
        'var': data.var(axis=1),
        'median': data.median(axis=1),
        'min': data.min(axis=1),
        'max': data.max(axis=1),
        'range': data.max(axis=1) - data.min(axis=1)
    })
    return features

# 1. MIT-BIH Dataset (No Resampling)
y_train_mitbih = mitbih_train.iloc[:, -1].values
y_test_mitbih = mitbih_test.iloc[:, -1].values
X_train_mitbih = mitbih_train.iloc[:, :-1]
X_test_mitbih = mitbih_test.iloc[:, :-1]

# Extract features for KNN classifier
train_features_mitbih = extract_features(X_train_mitbih)
test_features_mitbih = extract_features(X_test_mitbih)

# Train and evaluate KNN for MIT-BIH dataset
classifier_mitbih = KNeighborsClassifier(n_neighbors=10)
classifier_mitbih.fit(train_features_mitbih, y_train_mitbih)
predictions_mitbih = classifier_mitbih.predict(test_features_mitbih)

# Evaluate MIT-BIH performance
accuracy_mitbih = accuracy_score(y_test_mitbih, predictions_mitbih)
f1_mitbih = f1_score(y_test_mitbih, predictions_mitbih, average='weighted')
precision_mitbih = precision_score(y_test_mitbih, predictions_mitbih, average='weighted')
recall_mitbih = recall_score(y_test_mitbih, predictions_mitbih, average='weighted')
conf_matrix_mitbih = confusion_matrix(y_test_mitbih, predictions_mitbih)

# Print MIT-BIH results
print("\nMITBIH Dataset Results:")
print(f"  - Accuracy: {accuracy_mitbih:.4f}")
print(f"  - F1 Score: {f1_mitbih:.4f}")
print(f"  - Precision: {precision_mitbih:.4f}")
print(f"  - Recall: {recall_mitbih:.4f}")
print("  - Confusion Matrix:\n", conf_matrix_mitbih)

# 2. PTBDB Dataset (No Resampling)
ptbdb_abnormal['label'] = 1  # Label 1 for abnormal
ptbdb_normal['label'] = 0    # Label 0 for normal

# Concatenate both datasets
data_ptbdb = pd.concat([ptbdb_abnormal, ptbdb_normal], axis=0).reset_index(drop=True)

# Split into features and labels
X_ptbdb = data_ptbdb.iloc[:, :-1]
y_ptbdb = data_ptbdb['label']

# Split into train and test sets
X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(
    X_ptbdb, y_ptbdb, test_size=0.2, random_state=42, stratify=y_ptbdb
)

# Extract features for KNN classifier
train_features_ptbdb = extract_features(X_train_ptbdb)
test_features_ptbdb = extract_features(X_test_ptbdb)

# Train and evaluate KNN for PTBDB dataset
classifier_ptbdb = KNeighborsClassifier(n_neighbors=10)
classifier_ptbdb.fit(train_features_ptbdb, y_train_ptbdb)
predictions_ptbdb = classifier_ptbdb.predict(test_features_ptbdb)

# Evaluate PTBDB performance
accuracy_ptbdb = accuracy_score(y_test_ptbdb, predictions_ptbdb)
f1_ptbdb = f1_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
precision_ptbdb = precision_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
recall_ptbdb = recall_score(y_test_ptbdb, predictions_ptbdb, average='weighted')
conf_matrix_ptbdb = confusion_matrix(y_test_ptbdb, predictions_ptbdb)

# Print PTBDB results
print("\nPTBDB Dataset Results:")
print(f"  - Accuracy: {accuracy_ptbdb:.4f}")
print(f"  - F1 Score: {f1_ptbdb:.4f}")
print(f"  - Precision: {precision_ptbdb:.4f}")
print(f"  - Recall: {recall_ptbdb:.4f}")
print("  - Confusion Matrix:\n", conf_matrix_ptbdb)
