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
# df = pd.read_csv("/home/hamza/peptide/Datasets/cellpenetrating/cellpenetrating_peptide.csv")
# sequences = df["cellpenetrating_peptide"].tolist()
# labels = df["label"].tolist()  # Adjust if needed




# # Load ESM2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
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
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
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

# save_path = "cellpenetrating_peptide_best_cnn_model.pth"
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




#                     for the probabilities


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
# df = pd.read_csv("/home/hamza/peptide/Datasets/cellpenetrating/cellpenetrating_peptide.csv")
# sequences = df["cellpenetrating_peptide"].tolist()
# labels = df["label"].tolist()

# # Load ESM2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
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

# # Compute embeddings for all sequences
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

# # Convert list to tensor
# all_embeddings = torch.stack(all_embeddings)
# all_labels = torch.tensor(labels, dtype=torch.long)
# all_sequences = np.array(sequences)

# # Split into train, val, test (include sequences)
# X_train_val, X_test, y_train_val, y_test, seq_train_val, seq_test = train_test_split(
#     all_embeddings, all_labels, all_sequences,
#     test_size=0.2, stratify=all_labels, random_state=1
# )

# X_train, X_val, y_train, y_val, seq_train, seq_val = train_test_split(
#     X_train_val, y_train_val, seq_train_val,
#     test_size=0.1, stratify=y_train_val, random_state=1
# )

# # Dataset class
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# # Create datasets
# train_dataset = EmbeddingDataset(X_train, y_train)
# val_dataset = EmbeddingDataset(X_val, y_val)
# test_dataset = EmbeddingDataset(X_test, y_test)

# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Model
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

# model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)

# # Training and evaluation functions
# def train_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     all_preds, all_labels = [], []
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
#     running_loss = 0.0
#     all_preds, all_labels = [], []
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
#     running_loss = 0.0
#     all_preds, all_labels = [], []
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

# # Training
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

# num_epochs = 3000
# patience = 30
# best_val_loss = float('inf')
# epochs_no_improve = 0
# best_model_wts = None

# for epoch in range(num_epochs):
#     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#     val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
#     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_model_wts = model.state_dict()
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

# if best_model_wts is not None:
#     model.load_state_dict(best_model_wts)

# torch.save(model.state_dict(), "cellpenetrating_peptide_best_cnn_model.pth")
# print("Best model saved")

# # Testing
# test_loss, test_acc, test_labels, test_preds = test_model(model, test_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# cm = confusion_matrix(test_labels, test_preds)
# tn, fp, fn, tp = cm.ravel()
# mcc = matthews_corrcoef(test_labels, test_preds)
# sensitivity = recall_score(test_labels, test_preds)
# specificity = tn / (tn + fp)

# print(f"Sensitivity: {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"MCC: {mcc:.4f}")

# # Save test results into CSV
# out_dir = "/home/hamza/peptide/prediction_for_docking"
# os.makedirs(out_dir, exist_ok=True)
# results_df = pd.DataFrame({
#     "Peptide": seq_test,
#     "True_Label": test_labels,
#     "Predicted_Label": test_preds
# })
# out_path = os.path.join(out_dir, "cellpenetrating_test_predictions.csv")
# results_df.to_csv(out_path, index=False)
# print(f"Predictions saved to {out_path}")




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

# # ---------------------------
# # Reproducibility
# # ---------------------------
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

# # ---------------------------
# # Load data
# # ---------------------------
# df = pd.read_csv("/home/hamza/peptide/Datasets/cellpenetrating/cellpenetrating_peptide.csv")
# sequences = df["cellpenetrating_peptide"].tolist()
# labels = df["label"].tolist()

# # ---------------------------
# # Load pretrained ESM2 model
# # ---------------------------
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # ---------------------------
# # Extract residue embeddings
# # ---------------------------
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]  # remove CLS, EOS
#     return residue_embeddings.cpu()

# # ---------------------------
# # Compute embeddings for all sequences
# # ---------------------------
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
# all_sequences = np.array(sequences)

# # ---------------------------
# # Train / Val / Test split
# # ---------------------------
# X_train_val, X_test, y_train_val, y_test, seq_train_val, seq_test = train_test_split(
#     all_embeddings, all_labels, all_sequences,
#     test_size=0.2, stratify=all_labels, random_state=1
# )

# X_train, X_val, y_train, y_val, seq_train, seq_val = train_test_split(
#     X_train_val, y_train_val, seq_train_val,
#     test_size=0.1, stratify=y_train_val, random_state=1
# )

# # ---------------------------
# # Dataset class
# # ---------------------------
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# # ---------------------------
# # DataLoaders
# # ---------------------------
# train_dataset = EmbeddingDataset(X_train, y_train)
# val_dataset = EmbeddingDataset(X_val, y_val)
# test_dataset = EmbeddingDataset(X_test, y_test)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # ---------------------------
# # Model definition
# # ---------------------------
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
#         x = x.permute(0, 2, 1)              # (batch, features, seq_len)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.permute(0, 2, 1)              # (batch, seq_len, features)
#         _, h_n = self.gru(x)                # GRU hidden states
#         h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
#         h_n = torch.cat([h_n[0], h_n[1]], dim=1)  # concat forward + backward
#         x = self.dropout(h_n)
#         out = self.fc(x)
#         return out

# model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)

# # ---------------------------
# # Training & validation
# # ---------------------------
# def train_epoch(model, dataloader, criterion, optimizer, device):
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

# def validate_epoch(model, dataloader, criterion, device):
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

# # ---------------------------
# # Testing (with probabilities)
# # ---------------------------
# def test_model(model, dataloader, criterion, device):
#     model.eval()
#     running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
#     softmax = nn.Softmax(dim=1)
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             running_loss += loss.item() * X.size(0)
#             probs = softmax(outputs)
#             preds = torch.argmax(probs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())
#     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds, np.array(all_probs)

# # ---------------------------
# # Train loop with early stopping
# # ---------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)

# num_epochs = 3000
# patience = 30
# best_val_loss = float("inf")
# epochs_no_improve = 0
# best_model_wts = None

# for epoch in range(num_epochs):
#     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#     val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
#     print(f"Epoch {epoch+1}/{num_epochs} | "
#           f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
#           f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_model_wts = model.state_dict()
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

# if best_model_wts is not None:
#     model.load_state_dict(best_model_wts)

# torch.save(model.state_dict(), "cellpenetrating_peptide_best_cnn_model.pth")
# print("Best model saved")

# # ---------------------------
# # Final testing
# # ---------------------------
# test_loss, test_acc, test_labels, test_preds, test_probs = test_model(model, test_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# cm = confusion_matrix(test_labels, test_preds)
# tn, fp, fn, tp = cm.ravel()
# mcc = matthews_corrcoef(test_labels, test_preds)
# sensitivity = recall_score(test_labels, test_preds)
# specificity = tn / (tn + fp)

# print(f"Sensitivity: {sensitivity:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"MCC: {mcc:.4f}")

# # ---------------------------
# # Save predictions with probabilities
# # ---------------------------
# out_dir = "/home/hamza/peptide/prediction_for_docking"
# os.makedirs(out_dir, exist_ok=True)

# results_df = pd.DataFrame({
#     "Peptide": seq_test,
#     "True_Label": test_labels,
#     "Predicted_Label": test_preds,
#     "Prob_Class0": test_probs[:, 0],
#     "Prob_Class1": test_probs[:, 1]
# })

# out_path = os.path.join(out_dir, "cellpenetrating_test_predictions.csv")
# results_df.to_csv(out_path, index=False)

# print(f"Results with probabilities saved to {out_path}")







# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef
# import esm

# # ------------------ Seed and Device Setup ------------------
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
# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")



# # ------------------ Load Dataset ------------------
# df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/cellpenetrating/cellpenetrating_peptide.csv")
# sequences = df["cellpenetrating_peptide"].tolist()
# labels = df["label"].tolist()

# # ------------------ ESM2 Embeddings ------------------
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model = model.to(device)
# model.eval()

# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[6])
#     token_representations = results["representations"][6]
#     residue_embeddings = token_representations[0, 1:-1]
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

# # ------------------ Independent Test Split ------------------
# X_train_val, X_test, y_train_val, y_test = train_test_split(
#     all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42
# )

# # ------------------ Dataset Class ------------------
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels
#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# # ------------------ Model Definition ------------------
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

# # ------------------ Helper Functions ------------------
# def train_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0
#     preds_all, labels_all = [], []
#     for X, y in dataloader:
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         out = model(X)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * X.size(0)
#         preds = torch.argmax(out, 1)
#         preds_all.extend(preds.cpu().numpy())
#         labels_all.extend(y.cpu().numpy())
#     return running_loss / len(dataloader.dataset), accuracy_score(labels_all, preds_all)

# def validate_epoch(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0
#     preds_all, labels_all = [], []
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             out = model(X)
#             loss = criterion(out, y)
#             running_loss += loss.item() * X.size(0)
#             preds = torch.argmax(out, 1)
#             preds_all.extend(preds.cpu().numpy())
#             labels_all.extend(y.cpu().numpy())
#     return running_loss / len(dataloader.dataset), accuracy_score(labels_all, preds_all), labels_all, preds_all

# # ------------------ 10-Fold Cross Validation ------------------
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# criterion = nn.CrossEntropyLoss()
# num_epochs = 300
# patience = 30
# fold_results = []
# best_model_state = None
# best_acc = 0



# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_val, y_train_val)):
#     print(f"\n=== Fold {fold+1} ===")
#     train_data = EmbeddingDataset(X_train_val[train_idx], y_train_val[train_idx])
#     val_data = EmbeddingDataset(X_train_val[val_idx], y_train_val[val_idx])
#     train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

#     model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
#     optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)
#     best_loss, no_improve = float('inf'), 0

#     for epoch in range(num_epochs):
#         tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_acc, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)
#         print(f"Epoch {epoch+1}: Train {tr_acc:.4f}, Val {val_acc:.4f}")
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_state = model.state_dict()
#             no_improve = 0
#         else:
#             no_improve += 1
#             if no_improve >= patience:
#                 print("Early stopping.")
#                 break

#     model.load_state_dict(best_state)
#     mcc = matthews_corrcoef(y_true, y_pred)
#     rec = recall_score(y_true, y_pred)
#     cm = confusion_matrix(y_true, y_pred)
#     tn, fp, fn, tp = cm.ravel()
#     spec = tn / (tn + fp)
#     print(f"Fold {fold+1}: Acc {val_acc:.4f}, Recall {rec:.4f}, Spec {spec:.4f}, MCC {mcc:.4f}")

#     fold_results.append({"fold": fold+1, "acc": val_acc, "rec": rec, "spec": spec, "mcc": mcc})
#     if val_acc > best_acc:
#         best_acc = val_acc
#         best_model_state = best_state

# # ------------------ Test Evaluation ------------------
# best_model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
# best_model.load_state_dict(best_model_state)

# test_data = EmbeddingDataset(X_test, y_test)
# test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# test_model_loss = 0
# all_preds, all_labels = [], []
# best_model.eval()
# with torch.no_grad():
#     for X, y in test_loader:
#         X, y = X.to(device), y.to(device)
#         out = best_model(X)
#         loss = criterion(out, y)
#         test_model_loss += loss.item() * X.size(0)
#         preds = torch.argmax(out, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(y.cpu().numpy())

# test_loss = test_model_loss / len(test_loader.dataset)
# test_acc = accuracy_score(all_labels, all_preds)
# rec = recall_score(all_labels, all_preds)
# cm = confusion_matrix(all_labels, all_preds)
# tn, fp, fn, tp = cm.ravel()
# spec = tn / (tn + fp)
# mcc = matthews_corrcoef(all_labels, all_preds)

# print("\n=== Independent Test Results ===")
# print(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Recall: {rec:.4f}, Spec: {spec:.4f}, MCC: {mcc:.4f}")

# # ------------------ Save Results ------------------
# out_dir = "/home/hamza/peptide/prediction_for_docking"
# os.makedirs(out_dir, exist_ok=True)
# torch.save(best_model.state_dict(), os.path.join(out_dir, "cellpenetrating_best_fold_model.pth"))

# pd.DataFrame({
#     "True_Label": all_labels,
#     "Predicted_Label": all_preds
# }).to_csv(os.path.join(out_dir, "cellpenetrating_test_predictions.csv"), index=False)

# pd.DataFrame(fold_results).to_csv(os.path.join(out_dir, "cellpenetrating_fold_results.csv"), index=False)
# print("All results saved.")


  # updated code having the means etc ..
  
  
  
  
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

# ------------------ Seed and Device Setup ------------------
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

# ------------------ Load Dataset ------------------
df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/cellpenetrating/cellpenetrating_peptide.csv")
sequences = df["cellpenetrating_peptide"].tolist()
labels = df["label"].tolist()

# ------------------ ESM2 Embeddings ------------------
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

# ------------------ Prepare Embeddings ------------------
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

# ------------------ Dataset Class ------------------
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

# ------------------ Train/Validation Helpers ------------------
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

# ------------------ 10-Fold Cross Validation ------------------
# ------------------ 10-Fold Cross Validation ------------------
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

# ------------------ Compute Mean ± Std ------------------
fold_df = pd.DataFrame(fold_results)
mean_std = fold_df.mean().to_dict()
std_std = fold_df.std().to_dict()
print("\n=== 10-Fold Summary ===")
print(f"Acc: {mean_std['acc']:.4f} ± {std_std['acc']:.4f}")
print(f"Recall: {mean_std['rec']:.4f} ± {std_std['rec']:.4f}")
print(f"Spec: {mean_std['spec']:.4f} ± {std_std['spec']:.4f}")
print(f"MCC: {mean_std['mcc']:.4f} ± {std_std['mcc']:.4f}")

# ------------------ Independent Test Evaluation ------------------
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
  
  
  
  
  
  
  
  
  