import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
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

# Process MITBIH Dataset
y_train_mitbih = mitbih_train.iloc[:, -1].values
y_test_mitbih = mitbih_test.iloc[:, -1].values
X_train_mitbih = mitbih_train.iloc[:, :-1].values
X_test_mitbih = mitbih_test.iloc[:, :-1].values

# Split into training, validation, and test sets for MITBIH
X_train_mitbih, X_val_mitbih, y_train_mitbih, y_val_mitbih = train_test_split(
    X_train_mitbih, y_train_mitbih, test_size=0.2, random_state=42, stratify=y_train_mitbih
)
X_train_mitbih_tensor = torch.tensor(X_train_mitbih, dtype=torch.float32).unsqueeze(1).to(device)
y_train_mitbih_tensor = torch.tensor(y_train_mitbih, dtype=torch.long).to(device)
X_val_mitbih_tensor = torch.tensor(X_val_mitbih, dtype=torch.float32).unsqueeze(1).to(device)
y_val_mitbih_tensor = torch.tensor(y_val_mitbih, dtype=torch.long).to(device)
X_test_mitbih_tensor = torch.tensor(X_test_mitbih, dtype=torch.float32).unsqueeze(1).to(device)
y_test_mitbih_tensor = torch.tensor(y_test_mitbih, dtype=torch.long).to(device)

train_mitbih_data = TensorDataset(X_train_mitbih_tensor, y_train_mitbih_tensor)
val_mitbih_data = TensorDataset(X_val_mitbih_tensor, y_val_mitbih_tensor)
test_mitbih_data = TensorDataset(X_test_mitbih_tensor, y_test_mitbih_tensor)
train_loader_mitbih = DataLoader(train_mitbih_data, batch_size=64, shuffle=True)
val_loader_mitbih = DataLoader(val_mitbih_data, batch_size=32, shuffle=False)
test_loader_mitbih = DataLoader(test_mitbih_data, batch_size=32, shuffle=False)

# Process PTBDB Dataset
ptbdb_abnormal.columns = range(ptbdb_abnormal.shape[1])
ptbdb_normal.columns = range(ptbdb_normal.shape[1])
ptbdb_combined = pd.concat([ptbdb_abnormal, ptbdb_normal], ignore_index=True)
X_ptbdb = ptbdb_combined.iloc[:, :-1].values
y_ptbdb = ptbdb_combined.iloc[:, -1].values

# Split into training, validation, and test sets for PTBDB
X_train_ptbdb, X_val_ptbdb, y_train_ptbdb, y_val_ptbdb = train_test_split(
    X_ptbdb, y_ptbdb, test_size=0.2, random_state=42, stratify=y_ptbdb
)
X_train_ptbdb_tensor = torch.tensor(X_train_ptbdb, dtype=torch.float32).unsqueeze(1).to(device)
y_train_ptbdb_tensor = torch.tensor(y_train_ptbdb, dtype=torch.long).to(device)
X_val_ptbdb_tensor = torch.tensor(X_val_ptbdb, dtype=torch.float32).unsqueeze(1).to(device)
y_val_ptbdb_tensor = torch.tensor(y_val_ptbdb, dtype=torch.long).to(device)
X_test_ptbdb_tensor = torch.tensor(X_ptbdb, dtype=torch.float32).unsqueeze(1).to(device)
y_test_ptbdb_tensor = torch.tensor(y_ptbdb, dtype=torch.long).to(device)

train_ptbdb_data = TensorDataset(X_train_ptbdb_tensor, y_train_ptbdb_tensor)
val_ptbdb_data = TensorDataset(X_val_ptbdb_tensor, y_val_ptbdb_tensor)
test_ptbdb_data = TensorDataset(X_test_ptbdb_tensor, y_test_ptbdb_tensor)
train_loader_ptbdb = DataLoader(train_ptbdb_data, batch_size=64, shuffle=True)
val_loader_ptbdb = DataLoader(val_ptbdb_data, batch_size=32, shuffle=False)
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

# Training and evaluation function with validation
def train_and_evaluate(model, optimizer, train_loader, val_loader, test_loader, dataset_name):
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
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}] - {dataset_name} Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_{dataset_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} for {dataset_name}")
                break

    # Load best model for final evaluation on test set
    model.load_state_dict(torch.load(f"best_model_{dataset_name}.pth"))
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
train_and_evaluate(model_mitbih, optimizer_mitbih, train_loader_mitbih, val_loader_mitbih, test_loader_mitbih, "MITBIH")

print("\nTraining and evaluating on PTBDB dataset:")
train_and_evaluate(model_ptbdb, optimizer_ptbdb, train_loader_ptbdb, val_loader_ptbdb, test_loader_ptbdb, "PTBDB")
