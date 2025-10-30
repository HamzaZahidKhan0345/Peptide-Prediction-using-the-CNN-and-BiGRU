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


df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/antimicrobial/antimicrobial.csv")
sequences = df["antimicrobial_peptide"].tolist()
labels = df["label"].tolist()

# ---------------- Load ESM2 ----------------
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model_esm.eval()
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model_esm = model_esm.to(device)

def extract_residue_embeddings(seq):
    data = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[6])
    token_representations = results["representations"][6]
    residue_embeddings = token_representations[0, 1:-1]  # remove CLS and EOS
    return residue_embeddings.cpu()

all_embeddings = []
max_len = 50

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
    total_loss = 0
    preds, labels = [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total_loss / len(dataloader.dataset), accuracy_score(labels, preds)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return total_loss / len(dataloader.dataset), accuracy_score(labels, preds), labels, preds

criterion = nn.CrossEntropyLoss()
num_epochs = 3000
patience = 30
batch_size = 64
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []
best_fold_model = None
best_fold_acc = 0

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
    print(f"\n========== Fold {fold+1} ==========")
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
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Fold {fold+1}: Acc {val_acc:.4f}, Recall {recall:.4f}, Spec {specificity:.4f}, MCC {mcc:.4f}")
    fold_results.append({"Fold": fold+1, "Acc": val_acc, "Recall": recall, "Spec": specificity, "MCC": mcc})

    if val_acc > best_fold_acc:
        best_fold_acc = val_acc
        best_fold_model = model.state_dict()


df_folds = pd.DataFrame(fold_results)
mean_vals = df_folds.mean(numeric_only=True)
std_vals = df_folds.std(numeric_only=True)
print("\n===== 10-Fold Summary =====")
for metric in ["Acc", "Recall", "Spec", "MCC"]:
    print(f"{metric}: {mean_vals[metric]:.4f} ± {std_vals[metric]:.4f}")


save_dir = "/media/8TB_hardisk/hamza/peptide/updated_saved_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "antimicrobial.pt")
torch.save(best_fold_model, save_path)
print(f"\nBest fold model saved at: {save_path}")

print("\n===== Independent Test Evaluation =====")
model = CNN_BiGRU_Classifier().to(device)
model.load_state_dict(torch.load(save_path))
test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
recall = recall_score(all_labels, all_preds)
specificity = tn / (tn + fp)
mcc = matthews_corrcoef(all_labels, all_preds)

print(f"Test Acc: {test_acc:.4f}, Recall: {recall:.4f}, Spec: {specificity:.4f}, MCC: {mcc:.4f}")

out_dir = "/media/8TB_hardisk/hamza/peptide/updated_saved_models"
os.makedirs(out_dir, exist_ok=True)


pred_df = pd.DataFrame({"True_Label": all_labels, "Predicted_Label": all_preds})
pred_path = os.path.join(out_dir, "antimicrobial_test_predictions.csv")
pred_df.to_csv(pred_path, index=False)


fold_csv = os.path.join(out_dir, "antimicrobial_10fold_results.csv")
df_folds.to_csv(fold_csv, index=False)

print(f"\nTest predictions saved to: {pred_path}")
print(f"Cross-validation summary saved to: {fold_csv}")

