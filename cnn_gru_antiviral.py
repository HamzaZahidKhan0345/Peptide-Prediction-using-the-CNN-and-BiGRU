# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, matthews_corrcoef, recall_score,
#     confusion_matrix, f1_score
# )
# import esm

# # -------------------------
# # Set seed
# # -------------------------
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

# # -------------------------
# # Load data
# # -------------------------
# df = pd.read_csv("/home/hamza/peptide/Datasets/antiviral/antiviral.csv")
# sequences = df["antiviral_peptide"].tolist()
# labels = df["label"].tolist()

# # -------------------------
# # Load ESM2 model
# # -------------------------
# model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model_esm.eval()
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# model_esm = model_esm.to(device)

# # -------------------------
# # Function to extract residue embeddings
# # -------------------------
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     _, _, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model_esm(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]  # remove CLS and EOS
#     return residue_embeddings.cpu()

# # -------------------------
# # Precompute embeddings
# # -------------------------
# all_embeddings = []
# max_len = 50  # max peptide length (truncate/pad)

# for seq in sequences:
#     emb = extract_residue_embeddings(seq)
#     L, D = emb.shape
#     if L > max_len:
#         emb = emb[:max_len]
#     else:
#         pad = torch.zeros((max_len - L, D))
#         emb = torch.cat((emb, pad), dim=0)
#     all_embeddings.append(emb)

# all_embeddings = torch.stack(all_embeddings)  # (N, max_len, D)
# all_labels = torch.tensor(labels, dtype=torch.long)

# # -------------------------
# # Train / Val / Test Split
# # -------------------------
# X_train_val, X_test, y_train_val, y_test = train_test_split(
#     all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=1
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_val, y_train_val, test_size=0.1, stratify=y_train_val, random_state=1
# )

# # -------------------------
# # Dataset class
# # -------------------------
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# train_dataset = EmbeddingDataset(X_train, y_train)
# val_dataset = EmbeddingDataset(X_val, y_val)
# test_dataset = EmbeddingDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # -------------------------
# # CNN-BiGRU Classifier
# # -------------------------
# class CNN_BiGRU_Classifier(nn.Module):
#     def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.gru = nn.GRU(
#             input_size=128, hidden_size=hidden_dim, num_layers=gru_layers,
#             batch_first=True, bidirectional=True
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)

#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.permute(0, 2, 1)    # (batch, seq_len, 128)
#         _, h_n = self.gru(x)
#         h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
#         h_n = torch.cat([h_n[0], h_n[1]], dim=1)
#         x = self.dropout(h_n)
#         out = self.fc(x)
#         return out

# model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)

# # -------------------------
# # Evaluation with all metrics
# # -------------------------
# def evaluate_with_metrics(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             running_loss += loss.item() * X.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())

#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
#     epoch_loss = running_loss / len(dataloader.dataset)

#     acc = accuracy_score(all_labels, all_preds)
#     sensitivity = recall_score(all_labels, all_preds)
#     cm = confusion_matrix(all_labels, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / (tn + fp)
#     mcc = matthews_corrcoef(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds)

#     return {
#         "loss": epoch_loss,
#         "accuracy": acc,
#         "sensitivity": sensitivity,
#         "specificity": specificity,
#         "mcc": mcc,
#         "f1": f1
#     }

# # -------------------------
# # Training Loop with Early Stopping
# # -------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

# num_epochs = 3000
# patience = 30
# best_val_loss = float('inf')
# epochs_no_improve = 0
# best_model_wts = None

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     all_preds, all_labels = [], []
#     for X, y in train_loader:
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         outputs = model(X)
#         loss = criterion(outputs, y)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * X.size(0)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y.cpu().numpy())

#     train_loss = running_loss / len(train_loader.dataset)
#     train_acc = accuracy_score(all_labels, all_preds)

#     val_metrics = evaluate_with_metrics(model, val_loader, criterion, device)

#     print(
#         f"Epoch {epoch+1}/{num_epochs} "
#         f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
#         f"| Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
#     )

#     if val_metrics["loss"] < best_val_loss:
#         best_val_loss = val_metrics["loss"]
#         best_model_wts = model.state_dict()
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"Early stopping triggered at epoch {epoch+1}")
#             break

# if best_model_wts is not None:
#     model.load_state_dict(best_model_wts)

# save_path = "antiviral_best_cnn_model.pth"
# torch.save(model.state_dict(), save_path)
# print(f"Best model saved to {save_path}")

# # -------------------------
# # Final Validation + Test Evaluation
# # -------------------------
# val_metrics = evaluate_with_metrics(model, val_loader, criterion, device)
# print("\nFinal Validation Metrics:")
# for k, v in val_metrics.items():
#     print(f"{k}: {v:.4f}")

# test_metrics = evaluate_with_metrics(model, test_loader, criterion, device)
# print("\nFinal Test Metrics:")
# for k, v in test_metrics.items():
#     print(f"{k}: {v:.4f}")










# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, matthews_corrcoef, recall_score,
#     confusion_matrix, f1_score
# )
# import esm

# # -------------------------
# # Set seed
# # -------------------------
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

# # -------------------------
# # Load data
# # -------------------------
# df = pd.read_csv("/home/hamza/peptide/Datasets/antiviral/antiviral.csv")
# sequences = df["antiviral_peptide"].tolist()
# labels = df["label"].tolist()

# # -------------------------
# # Load ESM2 model
# # -------------------------
# model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model_esm.eval()
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# model_esm = model_esm.to(device)

# # -------------------------
# # Function to extract residue embeddings
# # -------------------------
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     _, _, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model_esm(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]  # remove CLS and EOS
#     return residue_embeddings.cpu()

# # -------------------------
# # Precompute embeddings
# # -------------------------
# all_embeddings = []
# max_len = 50
# for seq in sequences:
#     emb = extract_residue_embeddings(seq)
#     L, D = emb.shape
#     if L > max_len:
#         emb = emb[:max_len]
#     else:
#         pad = torch.zeros((max_len - L, D))
#         emb = torch.cat((emb, pad), dim=0)
#     all_embeddings.append(emb)

# all_embeddings = torch.stack(all_embeddings)  # (N, max_len, D)
# all_labels = torch.tensor(labels, dtype=torch.long)
# all_sequences = np.array(sequences)

# # -------------------------
# # Train / Val / Test Split
# # -------------------------
# X_train_val, X_test, y_train_val, y_test, seq_train_val, seq_test = train_test_split(
#     all_embeddings, all_labels, all_sequences,
#     test_size=0.2, stratify=all_labels, random_state=1
# )
# X_train, X_val, y_train, y_val, seq_train, seq_val = train_test_split(
#     X_train_val, y_train_val, seq_train_val,
#     test_size=0.1, stratify=y_train_val, random_state=1
# )

# # -------------------------
# # Dataset class
# # -------------------------
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# train_dataset = EmbeddingDataset(X_train, y_train)
# val_dataset = EmbeddingDataset(X_val, y_val)
# test_dataset = EmbeddingDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # -------------------------
# # CNN-BiGRU Classifier
# # -------------------------
# class CNN_BiGRU_Classifier(nn.Module):
#     def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.gru = nn.GRU(
#             input_size=128, hidden_size=hidden_dim, num_layers=gru_layers,
#             batch_first=True, bidirectional=True
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.permute(0, 2, 1)
#         _, h_n = self.gru(x)
#         h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
#         h_n = torch.cat([h_n[0], h_n[1]], dim=1)
#         x = self.dropout(h_n)
#         out = self.fc(x)
#         return out

# model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)

# # -------------------------
# # Evaluation with metrics
# # -------------------------
# def evaluate_with_metrics(model, dataloader, criterion, device, return_preds=False):
#     model.eval()
#     running_loss = 0.0
#     all_preds, all_labels, all_probs = [], [], []

#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             running_loss += loss.item() * X.size(0)
#             probs = torch.softmax(outputs, dim=1)   # (batch, 2)
#             preds = torch.argmax(probs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())   # keep both class probs

#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
#     all_probs = np.array(all_probs)
#     epoch_loss = running_loss / len(dataloader.dataset)

#     acc = accuracy_score(all_labels, all_preds)
#     sensitivity = recall_score(all_labels, all_preds)
#     cm = confusion_matrix(all_labels, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / (tn + fp)
#     mcc = matthews_corrcoef(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds)

#     metrics = {
#         "loss": epoch_loss,
#         "accuracy": acc,
#         "sensitivity": sensitivity,
#         "specificity": specificity,
#         "mcc": mcc,
#         "f1": f1
#     }

#     if return_preds:
#         return metrics, all_labels, all_preds, all_probs
#     return metrics

# # -------------------------
# # Training Loop with Early Stopping
# # -------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

# num_epochs = 3000
# patience = 30
# best_val_loss = float('inf')
# epochs_no_improve = 0
# best_model_wts = None

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     all_preds, all_labels = [], []
#     for X, y in train_loader:
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         outputs = model(X)
#         loss = criterion(outputs, y)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * X.size(0)
#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y.cpu().numpy())

#     train_loss = running_loss / len(train_loader.dataset)
#     train_acc = accuracy_score(all_labels, all_preds)

#     val_metrics = evaluate_with_metrics(model, val_loader, criterion, device)

#     print(
#         f"Epoch {epoch+1}/{num_epochs} "
#         f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
#         f"| Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
#     )

#     if val_metrics["loss"] < best_val_loss:
#         best_val_loss = val_metrics["loss"]
#         best_model_wts = model.state_dict()
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"Early stopping triggered at epoch {epoch+1}")
#             break

# if best_model_wts is not None:
#     model.load_state_dict(best_model_wts)

# save_path = "antiviral_best_cnn_model.pth"
# torch.save(model.state_dict(), save_path)
# print(f"Best model saved to {save_path}")

# # -------------------------
# # Final Validation + Test Evaluation
# # -------------------------
# val_metrics = evaluate_with_metrics(model, val_loader, criterion, device)
# print("\nFinal Validation Metrics:")
# for k, v in val_metrics.items():
#     print(f"{k}: {v:.4f}")

# test_metrics, test_labels, test_preds, test_probs = evaluate_with_metrics(
#     model, test_loader, criterion, device, return_preds=True
# )
# print("\nFinal Test Metrics:")
# for k, v in test_metrics.items():
#     print(f"{k}: {v:.4f}")

# # -------------------------
# # Save predictions with peptides
# # -------------------------
# out_dir = "/home/hamza/peptide/prediction_for_docking"
# os.makedirs(out_dir, exist_ok=True)

# results_df = pd.DataFrame({
#     "Peptide": seq_test,
#     "True_Label": test_labels,
#     "Predicted_Label": test_preds,
#     "Prob_Class0": test_probs[:, 0],
#     "Prob_Class1": test_probs[:, 1]
# })

# out_path = os.path.join(out_dir, "antiviral_test_predictions.csv")
# results_df.to_csv(out_path, index=False)
# print(f"Predictions with probabilities saved to {out_path}")




#      10 FOLDS 







#




# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, confusion_matrix, f1_score
# import esm

# # ======================================================
# # Set seed for reproducibility
# # ======================================================
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

# # ======================================================
# # Load dataset
# # ======================================================
# df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/antiviral/antiviral.csv")
# sequences = df["antiviral_peptide"].tolist()
# labels = df["label"].tolist()
# print(f"Loaded {len(sequences)} sequences.")


# # ======================================================
# # Load ESM2 model
# # ======================================================
# model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model_esm.eval()
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# model_esm = model_esm.to(device)

# # ======================================================
# # Extract ESM2 residue embeddings
# # ======================================================
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     _, _, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model_esm(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]  # remove CLS, EOS
#     return residue_embeddings.cpu()

# max_len = 50
# all_embeddings = []

# for seq in sequences:
#     emb = extract_residue_embeddings(seq)
#     L, D = emb.shape
#     if L > max_len:
#         emb = emb[:max_len]
#     else:
#         pad = torch.zeros((max_len - L, D))
#         emb = torch.cat((emb, pad), dim=0)
#     all_embeddings.append(emb)

# all_embeddings = torch.stack(all_embeddings)
# all_labels = torch.tensor(labels, dtype=torch.long)
# print(f"Embeddings shape: {all_embeddings.shape}")

# # ======================================================
# # Train/Test Split (80/20)
# # ======================================================
# X_trainval, X_test, y_trainval, y_test = train_test_split(
#     all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42
# )

# # ======================================================
# # Dataset class
# # ======================================================
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels
#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# # ======================================================
# # CNN-BiGRU Model
# # ======================================================
# class CNN_BiGRU_Classifier(nn.Module):
#     def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
#         super().__init__()
#         self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.gru = nn.GRU(
#             input_size=128,
#             hidden_size=hidden_dim,
#             num_layers=gru_layers,
#             batch_first=True,
#             bidirectional=True,
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.permute(0, 2, 1)
#         _, h_n = self.gru(x)
#         h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
#         h_n = torch.cat([h_n[0], h_n[1]], dim=1)
#         x = self.dropout(h_n)
#         return self.fc(x)

# # ======================================================
# # Evaluation metrics
# # ======================================================
# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     all_preds, all_labels = [], []
#     total_loss = 0.0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             total_loss += loss.item() * X.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
#     acc = accuracy_score(all_labels, all_preds)
#     recall = recall_score(all_labels, all_preds)
#     cm = confusion_matrix(all_labels, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / (tn + fp)
#     mcc = matthews_corrcoef(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds)
#     return {
#         "loss": total_loss / len(dataloader.dataset),
#         "acc": acc,
#         "recall": recall,
#         "specificity": specificity,
#         "mcc": mcc,
#         "f1": f1,
#     }

# # ======================================================
# # 10-Fold Cross-Validation
# # ======================================================
# criterion = nn.CrossEntropyLoss()
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# fold_results = []
# best_fold_acc = 0.0
# best_fold_model = None

# os.makedirs("saved_models_antiviral", exist_ok=True)

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval, y_trainval)):
#     print(f"\n========== Fold {fold+1} ==========")
#     X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
#     y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

#     train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=64, shuffle=True)
#     val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=64, shuffle=False)

#     model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
#     optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

#     best_val_loss = float("inf")
#     best_model_wts = None
#     patience = 20
#     no_improve = 0
#     num_epochs = 200

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         all_preds, all_labels = [], []
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * X.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())

#         train_loss = total_loss / len(train_loader.dataset)
#         train_acc = accuracy_score(all_labels, all_preds)
#         val_metrics = evaluate(model, val_loader, criterion, device)

#         print(
#             f"Epoch {epoch+1:03d} | Train Acc: {train_acc:.4f} | "
#             f"Val Acc: {val_metrics['acc']:.4f} | Val Loss: {val_metrics['loss']:.4f}"
#         )

#         if val_metrics["loss"] < best_val_loss:
#             best_val_loss = val_metrics["loss"]
#             best_model_wts = model.state_dict()
#             no_improve = 0
#         else:
#             no_improve += 1
#             if no_improve >= patience:
#                 print("Early stopping.")
#                 break

#     model.load_state_dict(best_model_wts)
#     val_metrics = evaluate(model, val_loader, criterion, device)
#     print(f"Fold {fold+1} Results: {val_metrics}")

#     fold_results.append(val_metrics)

#     if val_metrics["acc"] > best_fold_acc:
#         best_fold_acc = val_metrics["acc"]
#         best_fold_model = model.state_dict()

# # ======================================================
# # Save the best model
# # ======================================================
# best_model_path = os.path.join("saved_models_antiviral", "best_fold_model.pth")
# torch.save(best_fold_model, best_model_path)
# print(f"\nBest fold model saved to {best_model_path} with accuracy {best_fold_acc:.4f}")

# # ======================================================
# # Final Evaluation on Independent Test Set
# # ======================================================
# best_model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
# best_model.load_state_dict(torch.load(best_model_path))

# test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=64, shuffle=False)
# test_metrics = evaluate(best_model, test_loader, criterion, device)

# print("\n===== Independent Test Metrics =====")
# for k, v in test_metrics.items():
#     print(f"{k}: {v:.4f}")



#    updated code for the means and standard deviation




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






























