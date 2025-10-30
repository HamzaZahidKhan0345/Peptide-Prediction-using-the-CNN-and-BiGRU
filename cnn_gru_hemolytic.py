import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, confusion_matrix
import esm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# ====================== Load Dataset ======================
df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/hemolytic/hemolytic_peptide.csv")
sequences = df["hemolytic_peptide"].tolist()
labels = df["label"].tolist()
print(f"Loaded {len(sequences)} hemolytic sequences.")


model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model_esm.eval()
model_esm = model_esm.to(device)


def extract_residue_embeddings(seq):
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[6])
    token_representations = results["representations"][6]
    residue_embeddings = token_representations[0, 1:-1]
    return residue_embeddings.cpu()

max_len = 50
all_embeddings = []
for seq in sequences:
    emb = extract_residue_embeddings(seq)
    L, D = emb.shape
    if L > max_len:
        emb = emb[:max_len]
    else:
        pad = torch.zeros((max_len - L, D))
        emb = torch.cat((emb, pad), dim=0)
    all_embeddings.append(emb)

all_embeddings = torch.stack(all_embeddings)
all_labels = torch.tensor(labels, dtype=torch.long)
print(f"Embeddings shape: {all_embeddings.shape}")


X_trainval, X_test, y_trainval, y_test = train_test_split(
    all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
print(f"Train/Val samples: {len(X_trainval)}, Test samples: {len(X_test)}")

# ====================== Dataset Class ======================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class CNN_BiGRU_Classifier(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.gru = nn.GRU(128, hidden_dim, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)
        h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
        h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        x = self.dropout(h_n)
        return self.fc(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return total_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds

def test_model(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(all_labels, all_preds)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    return acc, recall, specificity, mcc


criterion = nn.CrossEntropyLoss()
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []
best_fold_model = None
best_fold_acc = 0.0
num_epochs = 3000
patience = 30

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval, y_trainval), start=1):
    print(f"\n========== Fold {fold} ==========")
    train_dataset = EmbeddingDataset(X_trainval[train_idx], y_trainval[train_idx])
    val_dataset = EmbeddingDataset(X_trainval[val_idx], y_trainval[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = CNN_BiGRU_Classifier().to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_labels, val_preds = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_model_wts)
    _, val_acc, val_labels, val_preds = validate_epoch(model, val_loader, criterion, device)

    cm = confusion_matrix(val_labels, val_preds)
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(val_labels, val_preds)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(val_labels, val_preds)

    print(f"Fold {fold} Results → Acc: {val_acc:.4f}, Recall: {recall:.4f}, Spec: {specificity:.4f}, MCC: {mcc:.4f}")
    fold_results.append({"Fold": fold, "Acc": val_acc, "Recall": recall, "Spec": specificity, "MCC": mcc})

    if val_acc > best_fold_acc:
        best_fold_acc = val_acc
        best_fold_model = best_model_wts

# ====================== Mean ± Std Summary ======================
df_folds = pd.DataFrame(fold_results)
mean_vals = df_folds.mean(numeric_only=True)
std_vals = df_folds.std(numeric_only=True)

print("\n===== 10-Fold Summary =====")
for metric in ["Acc", "Recall", "Spec", "MCC"]:
    print(f"{metric}: {mean_vals[metric]:.4f} ± {std_vals[metric]:.4f}")

# ====================== Evaluate Best Fold on Independent Test Set ======================
best_model = CNN_BiGRU_Classifier().to(device)
best_model.load_state_dict(best_fold_model)
test_dataset = EmbeddingDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_acc, recall, specificity, mcc = test_model(best_model, test_loader, criterion, device)

print("\n===== Independent Test Results =====")
print(f"Test Acc: {test_acc:.4f} | Recall: {recall:.4f} | Spec: {specificity:.4f} | MCC: {mcc:.4f}")

# ====================== Save Results ======================
SAVE_DIR = "/media/8TB_hardisk/hamza/peptide/updated_saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
best_model_path = os.path.join(SAVE_DIR, "hemolytic.pt")
torch.save(best_model.state_dict(), best_model_path)
print(f"Best fold model saved at {best_model_path}")

pred_path = "/media/8TB_hardisk/hamza/peptide/prediction_for_docking"
os.makedirs(pred_path, exist_ok=True)
df_folds.to_csv(os.path.join(pred_path, "hemolytic_10fold_results.csv"), index=False)

# Save test predictions
test_preds = []
best_model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        preds = torch.argmax(best_model(X), dim=1)
        test_preds.extend(preds.cpu().numpy())
pd.DataFrame({"True_Label": y_test.numpy(), "Predicted_Label": test_preds}).to_csv(
    os.path.join(pred_path, "hemolytic_test_predictions.csv"), index=False
)

print("\nAll results and predictions saved successfully.")



















