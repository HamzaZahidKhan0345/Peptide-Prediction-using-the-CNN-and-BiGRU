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

# -------------------- 1. Set Seed --------------------
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

# -------------------- 2. Load Dataset --------------------
DATA_PATH = "/media/8TB_hardisk/hamza/peptide/Datasets/antiviral/antiviral.csv"
df = pd.read_csv(DATA_PATH)
sequences = df["antiviral_peptide"].tolist()
labels = df["label"].astype(int).tolist()
print(f"Loaded {len(sequences)} antiviral sequences.")

# -------------------- 3. Load ESM2 Model --------------------
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model_esm.eval()
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
model_esm = model_esm.to(device)

# -------------------- 4. Extract Embeddings --------------------
MAX_LEN = 50

def extract_residue_embeddings(seq):
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[6])
    token_representations = results["representations"][6]
    residue_embeddings = token_representations[0, 1:-1]  # remove CLS and EOS
    return residue_embeddings.cpu()

all_embeddings = []
for seq in sequences:
    emb = extract_residue_embeddings(seq)
    L, D = emb.shape
    if L > MAX_LEN:
        emb = emb[:MAX_LEN]
    else:
        pad = torch.zeros((MAX_LEN - L, D))
        emb = torch.cat((emb, pad), dim=0)
    all_embeddings.append(emb)

all_embeddings = torch.stack(all_embeddings)
all_labels = torch.tensor(labels, dtype=torch.long)
print(f"Embeddings shape: {all_embeddings.shape}")

# -------------------- 5. Split Data (80% train-val, 20% independent test) --------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# -------------------- 6. Dataset Class --------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# -------------------- 7. CNN-BiGRU Model --------------------
class CNN_BiGRU_Classifier(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.gru = nn.GRU(128, hidden_dim, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, feat, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, seq_len, feat)
        _, h_n = self.gru(x)
        h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
        h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        x = self.dropout(h_n)
        return self.fc(x)

# -------------------- 8. Training Utilities --------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
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
    total_loss = 0
    all_preds, all_labels = [], []
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


# -------------------- 9. 10-Fold Cross Validation --------------------
criterion = nn.CrossEntropyLoss()
num_epochs = 3000
patience = 30
batch_size = 64
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []
best_fold_model_state = None
best_fold_acc = 0

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val), start=1):
    print(f"\n========== Fold {fold} ==========")
    train_ds = EmbeddingDataset(X_train_val[train_idx], y_train_val[train_idx])
    val_ds = EmbeddingDataset(X_train_val[val_idx], y_train_val[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNN_BiGRU_Classifier().to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)
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
    _, val_acc, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0

    print(f"Fold {fold}: Acc {val_acc:.4f}, Recall {recall:.4f}, Spec {specificity:.4f}, MCC {mcc:.4f}")
    fold_results.append({"Fold": fold, "Acc": val_acc, "Recall": recall, "Spec": specificity, "MCC": mcc})

    if val_acc > best_fold_acc:
        best_fold_acc = val_acc
        best_fold_model_state = model.state_dict()

# -------------------- 10. Fold Summary --------------------
df_folds = pd.DataFrame(fold_results)
mean_vals = df_folds.mean(numeric_only=True)
std_vals = df_folds.std(numeric_only=True)
print("\n===== 10-Fold Summary =====")
for metric in ["Acc", "Recall", "Spec", "MCC"]:
    print(f"{metric}: {mean_vals[metric]:.4f} ± {std_vals[metric]:.4f}")

# -------------------- 11. Save Best Fold Model --------------------
SAVE_DIR = "/media/8TB_hardisk/hamza/peptide/updated_saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
best_model_path = os.path.join(SAVE_DIR, "antiviral.pt")
torch.save(best_fold_model_state, best_model_path)
print(f"Best fold model saved at {best_model_path}")

# -------------------- 12. Independent Test Evaluation --------------------
test_ds = EmbeddingDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = CNN_BiGRU_Classifier().to(device)
model.load_state_dict(best_fold_model_state)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
recall = recall_score(all_labels, all_preds, zero_division=0)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
mcc = matthews_corrcoef(all_labels, all_preds)
test_acc = accuracy_score(all_labels, all_preds)

print("\n===== Independent Test Results =====")
print(f"Test Acc: {test_acc:.4f}, Recall: {recall:.4f}, Spec: {specificity:.4f}, MCC: {mcc:.4f}")

# -------------------- 13. Save Predictions and Fold Results --------------------
results_df = pd.DataFrame({"True_Label": all_labels, "Predicted_Label": all_preds})
pred_path = "/media/8TB_hardisk/hamza/peptide/prediction_for_docking"
os.makedirs(pred_path, exist_ok=True)
results_df.to_csv(os.path.join(pred_path, "antiviral_test_predictions.csv"), index=False)
df_folds.to_csv(os.path.join(pred_path, "antiviral_10fold_results.csv"), index=False)
print("Test predictions and 10-fold summary saved.")






























