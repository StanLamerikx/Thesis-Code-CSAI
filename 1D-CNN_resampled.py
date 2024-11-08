import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load datasets
ptbdb_abnormal = pd.read_csv('./ptbdb_abnormal.csv')
ptbdb_normal = pd.read_csv('./ptbdb_normal.csv')
mitbih_test = pd.read_csv('./mitbih_test.csv')
mitbih_train = pd.read_csv('./mitbih_train.csv')

# Resampling for MITBIH
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
X_train_mitbih_tensor = torch.tensor(X_train_mitbih, dtype=torch.float32).unsqueeze(1).to(device)
y_train_mitbih_tensor = torch.tensor(y_train_mitbih, dtype=torch.long).to(device)
X_test_mitbih_tensor = torch.tensor(X_test_mitbih, dtype=torch.float32).unsqueeze(1).to(device)
y_test_mitbih_tensor = torch.tensor(y_test_mitbih, dtype=torch.long).to(device)

train_mitbih_data = TensorDataset(X_train_mitbih_tensor, y_train_mitbih_tensor)
test_mitbih_data = TensorDataset(X_test_mitbih_tensor, y_test_mitbih_tensor)
train_loader_mitbih = DataLoader(train_mitbih_data, batch_size=64, shuffle=True)
test_loader_mitbih = DataLoader(test_mitbih_data, batch_size=32, shuffle=False)

# Resampling for PTBDB
ptbdb_abnormal.columns = range(ptbdb_abnormal.shape[1])
ptbdb_normal.columns = range(ptbdb_normal.shape[1])
ptbdb_combined = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True)
X_ptbdb = ptbdb_combined.iloc[:, :-1].values
y_ptbdb = ptbdb_combined.iloc[:, -1].values

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

X_train_ptbdb = df_ptbdb_resampled.iloc[:, :-1].values
y_train_ptbdb = df_ptbdb_resampled['label'].values
X_train_ptbdb_tensor = torch.tensor(X_train_ptbdb, dtype=torch.float32).unsqueeze(1).to(device)
y_train_ptbdb_tensor = torch.tensor(y_train_ptbdb, dtype=torch.long).to(device)
X_test_ptbdb_tensor = torch.tensor(X_test_ptbdb, dtype=torch.float32).unsqueeze(1).to(device)
y_test_ptbdb_tensor = torch.tensor(y_test_ptbdb, dtype=torch.long).to(device)

train_ptbdb_data = TensorDataset(X_train_ptbdb_tensor, y_train_ptbdb_tensor)
test_ptbdb_data = TensorDataset(X_test_ptbdb_tensor, y_test_ptbdb_tensor)
train_loader_ptbdb = DataLoader(train_ptbdb_data, batch_size=64, shuffle=True)
test_loader_ptbdb = DataLoader(test_ptbdb_data, batch_size=32, shuffle=False)

# Define the 1D CNN model with batch normalization and dropout
class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * (input_dim // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_dim = X_train_mitbih_tensor.size(2)
num_classes_mitbih = len(np.unique(y_train_mitbih))
num_classes_ptbdb = len(np.unique(y_train_ptbdb))

model_mitbih = CNN1D(input_dim, num_classes_mitbih).to(device)
model_ptbdb = CNN1D(input_dim, num_classes_ptbdb).to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer_mitbih = torch.optim.Adam(model_mitbih.parameters(), lr=0.001)
optimizer_ptbdb = torch.optim.Adam(model_ptbdb.parameters(), lr=0.001)

# Training and evaluation function
def train_and_evaluate(model, optimizer, train_loader, test_loader, y_test, dataset_name):
    num_epochs = 150
    early_stopping_patience = 40
    best_val_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Early stopping mechanism could be added here based on validation

    # Load best model and evaluate on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"\n{dataset_name} Test Results:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print("  - Confusion Matrix:\n", conf_matrix)

# Train and evaluate the model on both datasets
print("\nTraining and evaluating on MITBIH dataset:")
train_and_evaluate(model_mitbih, optimizer_mitbih, train_loader_mitbih, test_loader_mitbih, y_test_mitbih, "MITBIH")

print("\nTraining and evaluating on PTBDB dataset:")
train_and_evaluate(model_ptbdb, optimizer_ptbdb, train_loader_ptbdb, test_loader_ptbdb, y_test_ptbdb, "PTBDB")
