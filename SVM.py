import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

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

# Process PTBDB Dataset
ptbdb_abnormal.columns = range(ptbdb_abnormal.shape[1])
ptbdb_normal.columns = range(ptbdb_normal.shape[1])
ptbdb_combined = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True)
X_ptbdb = ptbdb_combined.iloc[:, :-1].values
y_ptbdb = ptbdb_combined.iloc[:, -1].values

# Split into training, validation, and test sets
X_train_ptbdb, X_val_ptbdb, y_train_ptbdb, y_val_ptbdb = train_test_split(
    X_ptbdb, y_ptbdb, test_size=0.2, random_state=42, stratify=y_ptbdb
)

# Flatten and standardize data for SVM
def preprocess_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_val_flat = scaler.transform(X_val.reshape(X_val.shape[0], -1))
    X_test_flat = scaler.transform(X_test.reshape(X_test.shape[0], -1))
    return X_train_flat, X_val_flat, X_test_flat

X_train_mitbih_flat, X_val_mitbih_flat, X_test_mitbih_flat = preprocess_data(X_train_mitbih, X_train_mitbih, X_test_mitbih)
X_train_ptbdb_flat, X_val_ptbdb_flat, X_test_ptbdb_flat = preprocess_data(X_train_ptbdb, X_val_ptbdb, X_ptbdb)

# Initialize and train SVM models
def train_and_evaluate_svm(X_train, y_train, X_test, y_test, dataset_name):
    svm_model = svm.SVC(kernel='rbf', C=1)
    print(f"Training SVM on {dataset_name} dataset...")
    svm_model.fit(X_train, y_train)
    
    # Evaluate SVM model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{dataset_name} SVM Test Results:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print("  - Confusion Matrix:\n", conf_matrix)

# Train and evaluate SVM for both datasets
train_and_evaluate_svm(X_train_mitbih_flat, y_train_mitbih, X_test_mitbih_flat, y_test_mitbih, "MITBIH")
train_and_evaluate_svm(X_train_ptbdb_flat, y_train_ptbdb, X_test_ptbdb_flat, y_ptbdb, "PTBDB")
