# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, confusion_matrix
# import esm


# # Set seed
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

# # Load data
# df = pd.read_csv("/home/hamza/peptide/Datasets/antimicrobial/antimicrobial.csv")
# sequences = df["antimicrobial_peptide"].tolist()
# labels = df["label"].tolist()  # Adjust if needed




# # Load ESM2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Function to extract residue embeddings
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]  # remove CLS and EOS tokens
#     return residue_embeddings.cpu()

# # Compute embeddings for all sequences once and store
# all_embeddings = []
# max_len = 50  # or whatever max length you want

# for seq in sequences:
#     emb = extract_residue_embeddings(seq)
#     L, D = emb.shape
#     if L > max_len:
#         emb = emb[:max_len]
#     else:
#         pad = torch.zeros((max_len - L, D))
#         emb = torch.cat((emb, pad), dim=0)
#     all_embeddings.append(emb)
    

# # Convert list to tensor for easier indexing
# all_embeddings = torch.stack(all_embeddings)  # shape (N, max_len, D)

# all_labels = torch.tensor(labels, dtype=torch.long)

# # Split data into train, val, test
# X_train_val, X_test, y_train_val, y_test = train_test_split(
#     all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=1)

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_val, y_train_val, test_size=0.1, stratify=y_train_val, random_state=1)

# # Define Dataset class using precomputed embeddings
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# # Create dataset objects
# train_dataset = EmbeddingDataset(X_train, y_train)
# val_dataset = EmbeddingDataset(X_val, y_val)
# test_dataset = EmbeddingDataset(X_test, y_test)

# # DataLoaders
# from torch.utils.data import DataLoader

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNN_BiGRU_Classifier(nn.Module):
#     def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
#         super().__init__()

#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

#         self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim, num_layers=gru_layers,
#                           batch_first=True, bidirectional=True)
        
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # bidirectional => 2*hidden_dim

#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         x = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_len) for Conv1d
#         x = F.relu(self.conv1(x))  # (batch, 128, seq_len)
#         x = F.relu(self.conv2(x))  # (batch, 256, seq_len)

#         x = x.permute(0, 2, 1)  # -> (batch, seq_len, 256) for GRU
#         _, h_n = self.gru(x)     # h_n: (num_layers*2, batch, hidden_dim)

#         h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]  # take last layer
#         h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 2*hidden_dim)

#         x = self.dropout(h_n)
#         out = self.fc(x)   
#         #print(out)# (batch, num_classes)
#         return out
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)

# import torch
# import torch.nn as nn
# from sklearn.metrics import accuracy_score

# def train_epoch(model, dataloader, criterion, optimizer, device): 
#     model.train()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     for X, y in dataloader:
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

#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     return epoch_loss, epoch_acc


# def validate_epoch(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             #print(y)
#             outputs = model(X)
#             #print(outputs)
#             loss = criterion(outputs, y)
            
#             running_loss += loss.item() * X.size(0)

#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())

#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     return epoch_loss, epoch_acc


# def test_model(model, dataloader, criterion, device):
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

#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     return epoch_loss, epoch_acc, all_labels, all_preds


# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5) 
   
# num_epochs = 3000
# patience = 30  # epochs to wait for improvement before stopping
# best_val_loss = float('inf')
# epochs_no_improve = 0
# best_model_wts = None

# train_losses = []
# train_accs = []
# val_losses = []
# val_accs = []

# for epoch in range(num_epochs):
#     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#     val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

#     train_losses.append(train_loss)
#     train_accs.append(train_acc)
#     val_losses.append(val_loss)
#     val_accs.append(val_acc)

#     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

#     # Early stopping check
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_model_wts = model.state_dict()
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"Early stopping triggered at epoch {epoch+1}")
#             break

# # Load best weights
# if best_model_wts is not None:
#     model.load_state_dict(best_model_wts)

# save_path = "antimicrobial_best_cnn_model.pth"
# torch.save(model.state_dict(), save_path)
# print(f"Best model saved to {save_path}")


# # After training, test evaluation:
# test_loss, test_acc, test_labels, test_preds = test_model(model, test_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# from sklearn.metrics import matthews_corrcoef, recall_score, confusion_matrix

# # Assuming test_labels and test_preds are available (lists or numpy arrays)
# cm = confusion_matrix(test_labels, test_preds)
# tn, fp, fn, tp = cm.ravel()

# mcc = matthews_corrcoef(test_labels, test_preds)
# sensitivity = recall_score(test_labels, test_preds)  # TP / (TP + FN)
# specificity = tn / (tn + fp)

# print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
# print(f"Sensitivity (Recall): {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")


#                             for probabilities
# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, confusion_matrix
# import esm

# # ============================================================
# # 1. Reproducibility
# # ============================================================
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

# # ============================================================
# # 2. Load Data
# # ============================================================
# df = pd.read_csv("/home/hamza/peptide/Datasets/antimicrobial/antimicrobial.csv")
# sequences = df["antimicrobial_peptide"].tolist()
# labels = df["label"].tolist()

# # ============================================================
# # 3. Load ESM2 Model
# # ============================================================
# model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model_esm.eval()
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model_esm = model_esm.to(device)

# # ============================================================
# # 4. Extract Residue Embeddings
# # ============================================================
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     _, _, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model_esm(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]  # remove CLS and EOS
#     return residue_embeddings.cpu()

# # ============================================================
# # 5. Precompute Embeddings
# # ============================================================
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

# # ============================================================
# # 6. Train / Val / Test Split
# # ============================================================
# X_train_val, X_test, y_train_val, y_test, seq_train_val, seq_test = train_test_split(
#     all_embeddings, all_labels, all_sequences,
#     test_size=0.2, stratify=all_labels, random_state=1
# )

# X_train, X_val, y_train, y_val, seq_train, seq_val = train_test_split(
#     X_train_val, y_train_val, seq_train_val,
#     test_size=0.1, stratify=y_train_val, random_state=1
# )

# # ============================================================
# # 7. Dataset Class
# # ============================================================
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

# # ============================================================
# # 8. CNN-BiGRU Classifier
# # ============================================================
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
#         x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
#         _, h_n = self.gru(x)
#         h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
#         h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # concat forward + backward
#         x = self.dropout(h_n)
#         out = self.fc(x)
#         return out

# model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)

# # ============================================================
# # 9. Training and Validation
# # ============================================================
# def train_epoch(model, dataloader, criterion, optimizer):
#     model.train()
#     running_loss, all_preds, all_labels = 0.0, [], []
#     for X, y in dataloader:
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
#     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

# def validate_epoch(model, dataloader, criterion):
#     model.eval()
#     running_loss, all_preds, all_labels = 0.0, [], []
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             running_loss += loss.item() * X.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

# # ============================================================
# # 10. Test with Probabilities
# # ============================================================
# def test_model(model, dataloader, criterion):
#     model.eval()
#     running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             running_loss += loss.item() * X.size(0)
#             probs = F.softmax(outputs, dim=1)
#             preds = torch.argmax(probs, dim=1)
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#     acc = accuracy_score(all_labels, all_preds)
#     return running_loss / len(dataloader.dataset), acc, all_labels, all_preds, np.array(all_probs)

# # ============================================================
# # 11. Training Loop with Early Stopping
# # ============================================================
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

# num_epochs, patience = 3000, 30
# best_val_loss, epochs_no_improve, best_model_wts = float('inf'), 0, None

# for epoch in range(num_epochs):
#     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
#     val_loss, val_acc = validate_epoch(model, val_loader, criterion)
#     print(f"Epoch {epoch+1}/{num_epochs} | "
#           f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
#           f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
#     if val_loss < best_val_loss:
#         best_val_loss, best_model_wts, epochs_no_improve = val_loss, model.state_dict(), 0
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

# if best_model_wts is not None:
#     model.load_state_dict(best_model_wts)

# torch.save(model.state_dict(), "antimicrobial_best_cnn_model.pth")
# print("Best model saved to antimicrobial_best_cnn_model.pth")

# # ============================================================
# # 12. Final Evaluation on Test Set
# # ============================================================
# test_loss, test_acc, test_labels, test_preds, test_probs = test_model(model, test_loader, criterion)

# print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# cm = confusion_matrix(test_labels, test_preds)
# tn, fp, fn, tp = cm.ravel()
# mcc = matthews_corrcoef(test_labels, test_preds)
# sensitivity = recall_score(test_labels, test_preds)
# specificity = tn / (tn + fp)

# print(f"Sensitivity: {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"MCC: {mcc:.4f}")

# # ============================================================
# # 13. Save Predictions with Probabilities
# # ============================================================
# out_dir = "/home/hamza/peptide/prediction_for_docking"
# os.makedirs(out_dir, exist_ok=True)

# results_df = pd.DataFrame({
#     "Peptide": seq_test,
#     "True_Label": test_labels,
#     "Predicted_Label": test_preds,
#     "Prob_Class0": test_probs[:, 0],
#     "Prob_Class1": test_probs[:, 1]
# })

# out_path = os.path.join(out_dir, "antimicrobial_test_predictions.csv")
# results_df.to_csv(out_path, index=False)
# print(f"Predictions with probabilities saved to {out_path}")

















# /media/8TB_hardisk/hamza/peptide/Datasets/antimicrobial/antimicrobial.csv




# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, confusion_matrix
# import esm

# # -------------------- 1. Set Seed --------------------
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

# # -------------------- 2. Load Dataset --------------------
# df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/antimicrobial/antimicrobial.csv")
# sequences = df["antimicrobial_peptide"].tolist()
# labels = df["label"].tolist()

# # -------------------- 3. Load ESM2 Model --------------------
# model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model_esm.eval()
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model_esm = model_esm.to(device)

# # -------------------- 4. Extract Embeddings --------------------
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model_esm(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]
#     return residue_embeddings.cpu()

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

# all_embeddings = torch.stack(all_embeddings)
# all_labels = torch.tensor(labels, dtype=torch.long)

# # -------------------- 5. Split Data (80% train-val, 20% independent test) --------------------
# X_train_val, X_test, y_train_val, y_test = train_test_split(
#     all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=1
# )

# # -------------------- 6. Define Dataset --------------------
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# # -------------------- 7. CNN-BiGRU Model --------------------
# class CNN_BiGRU_Classifier(nn.Module):
#     def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim, num_layers=gru_layers,
#                           batch_first=True, bidirectional=True)
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

# # -------------------- 8. Training Utilities --------------------
# def train_epoch(model, dataloader, criterion, optimizer, device): 
#     model.train()
#     running_loss, all_preds, all_labels = 0, [], []
#     for X, y in dataloader:
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
#     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

# def validate_epoch(model, dataloader, criterion, device):
#     model.eval()
#     running_loss, all_preds, all_labels = 0, [], []
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             running_loss += loss.item() * X.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

# def test_model(model, dataloader, criterion, device):
#     model.eval()
#     running_loss, all_preds, all_labels = 0, [], []
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             running_loss += loss.item() * X.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds

# # -------------------- 9. 5-Fold Cross Validation --------------------
# criterion = nn.CrossEntropyLoss()
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# best_val_acc = 0.0
# best_model_state = None

# for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
#     print(f"\nFold {fold+1}/10")
#     X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
#     y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

#     train_dataset = EmbeddingDataset(X_train, y_train)
#     val_dataset = EmbeddingDataset(X_val, y_val)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#     model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
#     optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

#     best_fold_val_acc = 0
#     patience, no_improve = 30, 0
#     best_fold_state = None

#     for epoch in range(100):  # shorter per fold
#         train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
#         print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

#         if val_acc > best_fold_val_acc:
#             best_fold_val_acc = val_acc
#             best_fold_state = model.state_dict()
#             no_improve = 0
#         else:
#             no_improve += 1
#             if no_improve >= patience:
#                 print("Early stopping triggered.")
#                 break

#     if best_fold_val_acc > best_val_acc:
#         best_val_acc = best_fold_val_acc
#         best_model_state = best_fold_state

# # -------------------- 10. Save Best Fold Model --------------------
# torch.save(best_model_state, "antimicrobial_best_cnn_model.pth")
# print(f"\nBest model from fold saved (Val Accuracy: {best_val_acc:.4f})")

# # -------------------- 11. Evaluate on Independent Test Set --------------------
# test_dataset = EmbeddingDataset(X_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# best_model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
# best_model.load_state_dict(torch.load("antimicrobial_best_cnn_model.pth"))

# test_loss, test_acc, test_labels, test_preds = test_model(best_model, test_loader, criterion, device)
# cm = confusion_matrix(test_labels, test_preds)
# tn, fp, fn, tp = cm.ravel()
# mcc = matthews_corrcoef(test_labels, test_preds)
# sensitivity = recall_score(test_labels, test_preds)
# specificity = tn / (tn + fp)

# print(f"\nIndependent Test Results:")
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Accuracy: {test_acc:.4f}")
# print(f"Sensitivity: {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"MCC: {mcc:.4f}")






#  updated code with standard deviation and folds results


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

# ---------------- Seed ----------------
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

# ---------------- Load data ----------------
df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/antimicrobial/antimicrobial.csv")
sequences = df["antimicrobial_peptide"].tolist()
labels = df["label"].tolist()

# ---------------- Load ESM2 ----------------
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model_esm.eval()
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model_esm = model_esm.to(device)

# ---------------- Embedding extraction ----------------
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

# ---------------- Split independent test set ----------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# ---------------- Dataset Class ----------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ---------------- Model ----------------
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

# ---------------- Train / Validate ----------------
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

# ---------------- 10-Fold Cross Validation ----------------
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

# ---------------- Summary: Mean ± Std ----------------
df_folds = pd.DataFrame(fold_results)
mean_vals = df_folds.mean(numeric_only=True)
std_vals = df_folds.std(numeric_only=True)
print("\n===== 10-Fold Summary =====")
for metric in ["Acc", "Recall", "Spec", "MCC"]:
    print(f"{metric}: {mean_vals[metric]:.4f} ± {std_vals[metric]:.4f}")

# ---------------- Save Best Model ----------------
save_dir = "/media/8TB_hardisk/hamza/peptide/updated_saved_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "antimicrobial.pt")
torch.save(best_fold_model, save_path)
print(f"\nBest fold model saved at: {save_path}")

# ---------------- Independent Test Evaluation ----------------
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

# ---------------- Save CSV Results ----------------
out_dir = "/media/8TB_hardisk/hamza/peptide/updated_saved_models"
os.makedirs(out_dir, exist_ok=True)

# Test predictions
pred_df = pd.DataFrame({"True_Label": all_labels, "Predicted_Label": all_preds})
pred_path = os.path.join(out_dir, "antimicrobial_test_predictions.csv")
pred_df.to_csv(pred_path, index=False)

# Fold summary
fold_csv = os.path.join(out_dir, "antimicrobial_10fold_results.csv")
df_folds.to_csv(fold_csv, index=False)

print(f"\nTest predictions saved to: {pred_path}")
print(f"Cross-validation summary saved to: {fold_csv}")

