import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef
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


df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/cellpenetrating/cellpenetrating_peptide.csv")
sequences = df["cellpenetrating_peptide"].tolist()
labels = df["label"].tolist()

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()

def extract_residue_embeddings(seq):
    data = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])
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

# ------------------ Independent Test Split ------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ------------------ Model Definition ------------------
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    preds_all, labels_all = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        preds_all.extend(torch.argmax(out, 1).cpu().numpy())
        labels_all.extend(y.cpu().numpy())
    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            running_loss += loss.item() * X.size(0)
            preds_all.extend(torch.argmax(out, 1).cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    return running_loss / len(loader.dataset), accuracy_score(labels_all, preds_all), labels_all, preds_all


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
criterion = nn.CrossEntropyLoss()
num_epochs = 3000
patience = 30
batch_size = 64

fold_results = []  # existing results from folds 1–8
best_model_state = None
best_acc = 0



for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
    

    print(f"\n=== Fold {fold+1} ===")

    train_data = EmbeddingDataset(X_train_val[train_idx], y_train_val[train_idx])
    val_data = EmbeddingDataset(X_train_val[val_idx], y_train_val[val_idx])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = CNN_BiGRU_Classifier().to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)
    best_loss, no_improve = float('inf'), 0

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train {tr_acc:.4f}, Val {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    mcc = matthews_corrcoef(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    print(f"Fold {fold+1}: Acc {val_acc:.4f}, Recall {rec:.4f}, Spec {spec:.4f}, MCC {mcc:.4f}")

    fold_results.append({"fold": fold+1, "acc": val_acc, "rec": rec, "spec": spec, "mcc": mcc})
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = best_state

fold_df = pd.DataFrame(fold_results)
mean_std = fold_df.mean().to_dict()
std_std = fold_df.std().to_dict()
print("\n=== 10-Fold Summary ===")
print(f"Acc: {mean_std['acc']:.4f} ± {std_std['acc']:.4f}")
print(f"Recall: {mean_std['rec']:.4f} ± {std_std['rec']:.4f}")
print(f"Spec: {mean_std['spec']:.4f} ± {std_std['spec']:.4f}")
print(f"MCC: {mean_std['mcc']:.4f} ± {std_std['mcc']:.4f}")


best_model = CNN_BiGRU_Classifier().to(device)
best_model.load_state_dict(best_model_state)
best_model.eval()

test_data = EmbeddingDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

all_preds, all_labels = [], []
test_loss = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        out = best_model(X)
        loss = criterion(out, y)
        test_loss += loss.item() * X.size(0)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

test_loss /= len(test_loader.dataset)
test_acc = accuracy_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp)
mcc = matthews_corrcoef(all_labels, all_preds)

print("\n=== Independent Test Results ===")
print(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Recall: {rec:.4f}, Spec: {spec:.4f}, MCC: {mcc:.4f}")

# ------------------ Save Models and Results ------------------
SAVE_DIR = "/media/8TB_hardisk/hamza/peptide/updated_saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(best_model_state, os.path.join(SAVE_DIR, "cellpenetrating.pt"))
print(f"Best fold model saved at {os.path.join(SAVE_DIR, 'cellpenetrating.pt')}")

PRED_DIR = "/media/8TB_hardisk/hamza/peptide/prediction_for_docking"
os.makedirs(PRED_DIR, exist_ok=True)
pd.DataFrame({"True_Label": all_labels, "Predicted_Label": all_preds}).to_csv(
    os.path.join(PRED_DIR, "cellpenetrating_test_predictions.csv"), index=False
)
pd.DataFrame(fold_results).to_csv(
    os.path.join(PRED_DIR, "cellpenetrating_10fold_results.csv"), index=False
)
print("All fold results and test predictions saved.")
  
  
  
  
  
  
  
  
  