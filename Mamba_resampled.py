import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load datasets
ptbdb_abnormal = pd.read_csv('./ptbdb_abnormal.csv')
ptbdb_normal = pd.read_csv('./ptbdb_normal.csv')
mitbih_test = pd.read_csv('./mitbih_test.csv')
mitbih_train = pd.read_csv('./mitbih_train.csv')

# MITBIH Dataset Processing with Resampling
y_train_mitbih = mitbih_train.iloc[:, -1].values
y_test_mitbih = mitbih_test.iloc[:, -1].values
X_train_mitbih = mitbih_train.iloc[:, :-1].values
X_test_mitbih = mitbih_test.iloc[:, :-1].values

# Resampling for MITBIH
df_mitbih_train = pd.DataFrame(X_train_mitbih)
df_mitbih_train['label'] = y_train_mitbih
df_0 = df_mitbih_train[df_mitbih_train['label'] == 0].sample(n=20000, random_state=42)
df_1 = resample(df_mitbih_train[df_mitbih_train['label'] == 1], replace=True, n_samples=20000, random_state=123)
df_2 = resample(df_mitbih_train[df_mitbih_train['label'] == 2], replace=True, n_samples=20000, random_state=124)
df_3 = resample(df_mitbih_train[df_mitbih_train['label'] == 3], replace=True, n_samples=20000, random_state=125)
df_4 = resample(df_mitbih_train[df_mitbih_train['label'] == 4], replace=True, n_samples=20000, random_state=126)
df_mitbih_resampled = pd.concat([df_0, df_1, df_2, df_3, df_4])

# Convert to tensors with correct shape
X_train_mitbih_tensor = torch.tensor(df_mitbih_resampled.iloc[:, :-1].values, dtype=torch.float32).unsqueeze(1).to(device)
y_train_mitbih_tensor = torch.tensor(df_mitbih_resampled['label'].values, dtype=torch.long).to(device)
X_test_mitbih_tensor = torch.tensor(X_test_mitbih, dtype=torch.float32).unsqueeze(1).to(device)
y_test_mitbih_tensor = torch.tensor(y_test_mitbih, dtype=torch.long).to(device)

# PTBDB Dataset Processing with Resampling
ptbdb_abnormal.columns = range(ptbdb_abnormal.shape[1])
ptbdb_normal.columns = range(ptbdb_normal.shape[1])
ptbdb_combined = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True)
X_ptbdb = ptbdb_combined.iloc[:, :-1].values
y_ptbdb = ptbdb_combined.iloc[:, -1].values

# Train-test split and resampling for PTBDB
X_train_ptbdb, X_test_ptbdb, y_train_ptbdb, y_test_ptbdb = train_test_split(
    X_ptbdb, y_ptbdb, test_size=0.2, random_state=42, stratify=y_ptbdb
)
df_ptbdb_train = pd.DataFrame(X_train_ptbdb)
df_ptbdb_train['label'] = y_train_ptbdb
df_normal = df_ptbdb_train[df_ptbdb_train['label'] == 0]
df_abnormal = df_ptbdb_train[df_ptbdb_train['label'] == 1]
sample_size = max(len(df_normal), len(df_abnormal))
df_normal_resampled = resample(df_normal, replace=True, n_samples=sample_size, random_state=42)
df_abnormal_resampled = resample(df_abnormal, replace=True, n_samples=sample_size, random_state=123)
df_ptbdb_resampled = pd.concat([df_normal_resampled, df_abnormal_resampled])

# Convert to tensors with correct shape
X_train_ptbdb_tensor = torch.tensor(df_ptbdb_resampled.iloc[:, :-1].values, dtype=torch.float32).unsqueeze(1).to(device)
y_train_ptbdb_tensor = torch.tensor(df_ptbdb_resampled['label'].values, dtype=torch.long).to(device)
X_test_ptbdb_tensor = torch.tensor(X_test_ptbdb, dtype=torch.float32).unsqueeze(1).to(device)
y_test_ptbdb_tensor = torch.tensor(y_test_ptbdb, dtype=torch.long).to(device)

# Define DataLoaders
train_loader_mitbih = DataLoader(TensorDataset(X_train_mitbih_tensor, y_train_mitbih_tensor), batch_size=64, shuffle=True)
test_loader_mitbih = DataLoader(TensorDataset(X_test_mitbih_tensor, y_test_mitbih_tensor), batch_size=64, shuffle=False)
train_loader_ptbdb = DataLoader(TensorDataset(X_train_ptbdb_tensor, y_train_ptbdb_tensor), batch_size=64, shuffle=True)
test_loader_ptbdb = DataLoader(TensorDataset(X_test_ptbdb_tensor, y_test_ptbdb_tensor), batch_size=64, shuffle=False)

# Model setup
model = Mamba(d_model=X_train_mitbih_tensor.size(2), d_state=16, d_conv=8, expand=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 150

# Training and evaluation
for dataset_name, train_loader, test_loader, y_test in [
    ("MITBIH", train_loader_mitbih, test_loader_mitbih, y_test_mitbih),
    ("PTBDB", train_loader_ptbdb, test_loader_ptbdb, y_test_ptbdb),
]:
    print(f"\nTraining on {dataset_name} dataset")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - {dataset_name} Training Loss: {running_loss / len(train_loader):.4f}")

    # Final evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze(1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Calculate test metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"\n{dataset_name} Test Results:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print("  - Confusion Matrix:\n", conf_matrix)
