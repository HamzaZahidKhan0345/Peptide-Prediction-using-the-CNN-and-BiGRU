#                       #     Actual code 


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
# df = pd.read_csv("/home/hamza/peptide/Datasets/hemolytic/hemolytic_peptide.csv")
# sequences = df["hemolytic_peptide"].tolist()
# labels = df["label"].tolist()  # Adjust if needed


# # Load ESM2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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

# save_path = "hemolytic_peptide_best_cnn_model.pth"
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







 
#                 #   code for the probability

# # import os
# # import random
# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import Dataset, DataLoader
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, confusion_matrix
# # import esm

# # # ---------------------------
# # # Set seed
# # # ---------------------------
# # def set_seed(seed=42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     os.environ['PYTHONHASHSEED'] = str(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # set_seed(42)

# # # ---------------------------
# # # Load data
# # # ---------------------------
# # df = pd.read_csv("/home/hamza/peptide/Datasets/hemolytic/hemolytic_peptide.csv")
# # sequences = df["hemolytic_peptide"].tolist()
# # labels = df["label"].tolist()

# # # ---------------------------
# # # Load ESM2 model
# # # ---------------------------
# # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# # batch_converter = alphabet.get_batch_converter()
# # model.eval()
# # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# # model = model.to(device)

# # # ---------------------------
# # # Function to extract residue embeddings
# # # ---------------------------
# # def extract_residue_embeddings(seq):
# #     data = [("protein", seq)]
# #     _, _, batch_tokens = batch_converter(data)
# #     batch_tokens = batch_tokens.to(device)
# #     with torch.no_grad():
# #         results = model(batch_tokens, repr_layers=[6])
# #     token_representations = results["representations"][6]
# #     residue_embeddings = token_representations[0, 1:-1]  # remove CLS and EOS
# #     return residue_embeddings.cpu()

# # # ---------------------------
# # # Compute embeddings
# # # ---------------------------
# # all_embeddings = []
# # max_len = 50
# # for seq in sequences:
# #     emb = extract_residue_embeddings(seq)
# #     L, D = emb.shape
# #     if L > max_len:
# #         emb = emb[:max_len]
# #     else:
# #         pad = torch.zeros((max_len - L, D))
# #         emb = torch.cat((emb, pad), dim=0)
# #     all_embeddings.append(emb)

# # all_embeddings = torch.stack(all_embeddings)
# # all_labels = torch.tensor(labels, dtype=torch.long)
# # all_sequences = np.array(sequences)

# # # ---------------------------
# # # Split data
# # # ---------------------------
# # X_train_val, X_test, y_train_val, y_test, seq_train_val, seq_test = train_test_split(
# #     all_embeddings, all_labels, all_sequences,
# #     test_size=0.2, stratify=all_labels, random_state=1
# # )

# # X_train, X_val, y_train, y_val, seq_train, seq_val = train_test_split(
# #     X_train_val, y_train_val, seq_train_val,
# #     test_size=0.1, stratify=y_train_val, random_state=1
# # )

# # # ---------------------------
# # # Dataset class
# # # ---------------------------
# # class EmbeddingDataset(Dataset):
# #     def __init__(self, embeddings, labels):
# #         self.embeddings = embeddings
# #         self.labels = labels

# #     def __len__(self):
# #         return len(self.labels)

# #     def __getitem__(self, idx):
# #         return self.embeddings[idx], self.labels[idx]

# # train_dataset = EmbeddingDataset(X_train, y_train)
# # val_dataset = EmbeddingDataset(X_val, y_val)
# # test_dataset = EmbeddingDataset(X_test, y_test)

# # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # # ---------------------------
# # # Model
# # # ---------------------------
# # class CNN_BiGRU_Classifier(nn.Module):
# #     def __init__(self, input_dim=1280, hidden_dim=128, gru_layers=1, num_classes=2, dropout=0.5):
# #         super().__init__()
# #         self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
# #         self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim,
# #                           num_layers=gru_layers, batch_first=True, bidirectional=True)
# #         self.dropout = nn.Dropout(dropout)
# #         self.fc = nn.Linear(hidden_dim * 2, num_classes)

# #     def forward(self, x):
# #         x = x.permute(0, 2, 1)
# #         x = F.relu(self.conv1(x))
# #         x = F.relu(self.conv2(x))
# #         x = x.permute(0, 2, 1)
# #         _, h_n = self.gru(x)
# #         h_n = h_n.view(self.gru.num_layers, 2, x.size(0), -1)[-1]
# #         h_n = torch.cat([h_n[0], h_n[1]], dim=1)
# #         x = self.dropout(h_n)
# #         return self.fc(x)

# # model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)

# # # ---------------------------
# # # Train and eval functions
# # # ---------------------------
# # def train_epoch(model, dataloader, criterion, optimizer, device): 
# #     model.train()
# #     running_loss, all_preds, all_labels = 0.0, [], []
# #     for X, y in dataloader:
# #         X, y = X.to(device), y.to(device)
# #         optimizer.zero_grad()
# #         outputs = model(X)
# #         loss = criterion(outputs, y)
# #         loss.backward()
# #         optimizer.step()
# #         running_loss += loss.item() * X.size(0)
# #         preds = torch.argmax(outputs, dim=1)
# #         all_preds.extend(preds.cpu().numpy())
# #         all_labels.extend(y.cpu().numpy())
# #     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

# # def validate_epoch(model, dataloader, criterion, device):
# #     model.eval()
# #     running_loss, all_preds, all_labels = 0.0, [], []
# #     with torch.no_grad():
# #         for X, y in dataloader:
# #             X, y = X.to(device), y.to(device)
# #             outputs = model(X)
# #             loss = criterion(outputs, y)
# #             running_loss += loss.item() * X.size(0)
# #             preds = torch.argmax(outputs, dim=1)
# #             all_preds.extend(preds.cpu().numpy())
# #             all_labels.extend(y.cpu().numpy())
# #     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds)

# # def test_model(model, dataloader, criterion, device):
# #     model.eval()
# #     running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
# #     softmax = nn.Softmax(dim=1)
# #     with torch.no_grad():
# #         for X, y in dataloader:
# #             X, y = X.to(device), y.to(device)
# #             outputs = model(X)
# #             loss = criterion(outputs, y)
# #             running_loss += loss.item() * X.size(0)
# #             probs = softmax(outputs)
# #             preds = torch.argmax(probs, dim=1)
# #             all_preds.extend(preds.cpu().numpy())
# #             all_labels.extend(y.cpu().numpy())
# #             all_probs.extend(probs.cpu().numpy())
# #     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds, np.array(all_probs)

# # # ---------------------------
# # # Training loop
# # # ---------------------------
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5) 
# # num_epochs, patience = 3000, 30
# # best_val_loss, epochs_no_improve, best_model_wts = float('inf'), 0, None

# # for epoch in range(num_epochs):
# #     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
# #     val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
# #     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
# #     if val_loss < best_val_loss:
# #         best_val_loss, best_model_wts, epochs_no_improve = val_loss, model.state_dict(), 0
# #     else:
# #         epochs_no_improve += 1
# #         if epochs_no_improve >= patience:
# #             print(f"Early stopping at epoch {epoch+1}")
# #             break

# # if best_model_wts is not None:
# #     model.load_state_dict(best_model_wts)

# # save_path = "hemolytic_peptide_best_cnn_model.pth"
# # torch.save(model.state_dict(), save_path)
# # print(f"Best model saved to {save_path}")

# # # ---------------------------
# # # Test evaluation
# # # ---------------------------
# # test_loss, test_acc, test_labels, test_preds, test_probs = test_model(model, test_loader, criterion, device)
# # print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# # cm = confusion_matrix(test_labels, test_preds)
# # tn, fp, fn, tp = cm.ravel()
# # mcc = matthews_corrcoef(test_labels, test_preds)
# # sensitivity = recall_score(test_labels, test_preds)
# # specificity = tn / (tn + fp)

# # print(f"Sensitivity: {sensitivity:.4f}")
# # print(f"Specificity: {specificity:.4f}")
# # print(f"MCC: {mcc:.4f}")

# # # ---------------------------
# # # Save peptides + labels + predictions + probabilities
# # # ---------------------------
# # out_dir = "/home/hamza/peptide/prediction_for_docking"
# # os.makedirs(out_dir, exist_ok=True)

# # results_df = pd.DataFrame({
# #     "Peptide": seq_test,
# #     "True_Label": test_labels,
# #     "Predicted_Label": test_preds,
# #     "Prob_Class0": test_probs[:, 0],
# #     "Prob_Class1": test_probs[:, 1]
# # })

# # out_path = os.path.join(out_dir, "hemolytic_test_predictions.csv")
# # results_df.to_csv(out_path, index=False)

# # print(f"Results with probabilities saved to {out_path}")







# code for the ten folds 



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

# # ====================== Seed Setting ======================
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

# # ====================== Load Dataset ======================
# df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/hemolytic/hemolytic_peptide.csv")
# sequences = df["hemolytic_peptide"].tolist()
# labels = df["label"].tolist()
# print(f"Loaded {len(sequences)} sequences.")

# # ====================== Load ESM2 Model ======================
# model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model_esm.eval()
# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# model_esm = model_esm.to(device)

# # ====================== Extract Embeddings ======================
# def extract_residue_embeddings(seq):
#     data = [("protein", seq)]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_tokens = batch_tokens.to(device)
#     with torch.no_grad():
#         results = model_esm(batch_tokens, repr_layers=[6])
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
# print(f"Embeddings shape: {all_embeddings.shape}")

# # ====================== Split Independent Test Set (20%) ======================
# X_trainval, X_test, y_trainval, y_test = train_test_split(
#     all_embeddings, all_labels, test_size=0.2, stratify=all_labels, random_state=42
# )
# print(f"Train/Val samples: {len(X_trainval)}, Test samples: {len(X_test)}")

# # ====================== Dataset Class ======================
# class EmbeddingDataset(Dataset):
#     def __init__(self, embeddings, labels):
#         self.embeddings = embeddings
#         self.labels = labels
#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

# # ====================== Model Definition ======================
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
#         return self.fc(x)

# # ====================== Helper Functions ======================
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
#     return running_loss / len(dataloader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds

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
#     epoch_loss = running_loss / len(dataloader.dataset)
#     acc = accuracy_score(all_labels, all_preds)
#     mcc = matthews_corrcoef(all_labels, all_preds)
#     recall = recall_score(all_labels, all_preds)
#     cm = confusion_matrix(all_labels, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / (tn + fp)
#     return epoch_loss, acc, recall, specificity, mcc

# # ====================== 10-Fold Cross Validation ======================
# criterion = nn.CrossEntropyLoss()
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# fold_results = []
# best_fold_model = None
# best_fold_acc = 0.0

# num_epochs = 200
# patience = 30

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval, y_trainval)):
#     print(f"\n========== Fold {fold+1} ==========")
    
#     train_dataset = EmbeddingDataset(X_trainval[train_idx], y_trainval[train_idx])
#     val_dataset = EmbeddingDataset(X_trainval[val_idx], y_trainval[val_idx])
    
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#     model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
#     optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)
    
#     best_val_loss = float('inf')
#     best_val_acc = 0.0
#     epochs_no_improve = 0
#     best_model_wts = None

#     for epoch in range(num_epochs):
#         train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_acc, val_labels, val_preds = validate_epoch(model, val_loader, criterion, device)
#         print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_val_acc = val_acc
#             best_model_wts = model.state_dict()
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print("Early stopping.")
#                 break

#     model.load_state_dict(best_model_wts)
#     cm = confusion_matrix(val_labels, val_preds)
#     tn, fp, fn, tp = cm.ravel()
#     mcc = matthews_corrcoef(val_labels, val_preds)
#     recall = recall_score(val_labels, val_preds)
#     specificity = tn / (tn + fp)

#     print(f"Fold {fold+1} Results: Val Acc: {best_val_acc:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, MCC: {mcc:.4f}")
#     fold_results.append({"fold": fold+1, "val_acc": best_val_acc, "recall": recall, "specificity": specificity, "mcc": mcc})

#     if best_val_acc > best_fold_acc:
#         best_fold_acc = best_val_acc
#         best_fold_model = best_model_wts

# # ====================== Evaluate Best Fold on Independent Test Set ======================
# print("\n===== Cross-Validation Summary =====")
# df_results = pd.DataFrame(fold_results)
# print(df_results)
# print("\nMean Performance Across 10 Folds:")
# print(df_results.mean(numeric_only=True))

# best_model = CNN_BiGRU_Classifier(input_dim=1280, num_classes=2, dropout=0.5).to(device)
# best_model.load_state_dict(best_fold_model)

# test_dataset = EmbeddingDataset(X_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# test_loss, test_acc, recall, specificity, mcc = test_model(best_model, test_loader, criterion, device)
# print("\n===== Independent Test Set Results =====")
# print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Recall: {recall:.4f} | Specificity: {specificity:.4f} | MCC: {mcc:.4f}")

# # ====================== Save Best Model and Results ======================
# save_model_path = "hemolytic_best_fold_model.pth"
# torch.save(best_model.state_dict(), save_model_path)
# print(f"\nBest fold model saved at: {save_model_path}")

# df_results.to_csv("hemolytic_crossval_results.csv", index=False)
# print("Fold-wise results saved to hemolytic_crossval_results.csv")





# updated code means standard deviation etc 

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

# ====================== Seed Setting ======================
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

# ====================== Device Setup ======================
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# ====================== Load Dataset ======================
df = pd.read_csv("/media/8TB_hardisk/hamza/peptide/Datasets/hemolytic/hemolytic_peptide.csv")
sequences = df["hemolytic_peptide"].tolist()
labels = df["label"].tolist()
print(f"Loaded {len(sequences)} hemolytic sequences.")

# ====================== Load ESM2 Model ======================
model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model_esm.eval()
model_esm = model_esm.to(device)

# ====================== Extract Embeddings ======================
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

# ====================== Split Independent Test Set (20%) ======================
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

# ====================== Model Definition ======================
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

# ====================== Helper Functions ======================
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

# ====================== 10-Fold Cross Validation ======================
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



















