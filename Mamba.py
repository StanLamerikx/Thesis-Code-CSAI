import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Early stopping parameters
early_stopping_patience = 40

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
train_loader_mitbih = DataLoader(train_mitbih_data, batch_size=132, shuffle=True)
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
train_loader_ptbdb = DataLoader(train_ptbdb_data, batch_size=32, shuffle=True)
val_loader_ptbdb = DataLoader(val_ptbdb_data, batch_size=32, shuffle=False)
test_loader_ptbdb = DataLoader(test_ptbdb_data, batch_size=32, shuffle=False)

# Initialize Mamba model
dim = X_train_mitbih_tensor.size(2)
model = Mamba(d_model=dim, d_state=32, d_conv=4, expand=2).to(device)

# Define optimizer and loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate with early stopping based on validation accuracy
num_epochs = 150
results = {"train_loss": [], "val_loss": [], "val_accuracy": []}

for dataset_name, train_loader, val_loader, test_loader in [
    ("MITBIH", train_loader_mitbih, val_loader_mitbih, test_loader_mitbih),
    ("PTBDB", train_loader_ptbdb, val_loader_ptbdb, test_loader_ptbdb),
]:

    print(f"Training on {dataset_name} dataset")
    
    # Reset early stopping variables for each dataset
    best_val_accuracy = 0.0
    patience_counter = 0

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

        # Calculate train, validation loss and accuracy
        train_loss = running_loss / len(train_loader)
        val_loss = 0.0
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze(1)
                val_loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        # Print loss and accuracy for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] - {dataset_name} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_{dataset_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} for {dataset_name} due to no improvement in accuracy")
                break

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["val_accuracy"].append(val_accuracy)

    # Evaluate on test set
    test_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze(1)
            test_loss += criterion(outputs, y_batch).item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    test_loss /= len(test_loader)

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

    # Plot training and validation loss, and validation accuracy
    plt.figure()
    plt.plot(results["train_loss"], label="Train Loss")
    plt.plot(results["val_loss"], label="Validation Loss")
    plt.plot(results["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"{dataset_name} Loss and Accuracy over Epochs")
    plt.legend()
    plt.savefig(f"{dataset_name}_metrics_plot.png")
    print(f"Saved {dataset_name} metrics plot.")
    plt.close()
