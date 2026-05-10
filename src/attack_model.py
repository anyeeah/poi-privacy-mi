import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


class AttackDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttackMLP(nn.Module):
    """
    Membership Inference Attack Model.
    
    Input : confidence score vector from Target Model (num_poi,)
    Output: membership probability (0=Non-member, 1=Member)
    """
    def __init__(self, input_dim=86):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def extract_confidence(model, X, device, batch_size=1024):
    """
    Extract confidence scores from Target Model.
    
    Args:
        model: trained POITransformer
        X: input sequences (n_samples, window_size)
        device: cuda or cpu
    
    Returns:
        confidence scores (n_samples, num_poi)
    """
    dataset = torch.tensor(X, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_conf = []

    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            prob = F.softmax(output, dim=1)
            all_conf.append(prob.cpu())

    return torch.cat(all_conf).numpy()


def train_attack(attack_model, train_loader, optimizer, criterion, device):
    attack_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = attack_model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()


def evaluate_attack(attack_model, test_loader, device):
    """
    Evaluate Attack Model.
    
    Returns:
        auc: AUC-ROC score
        acc: Attack Accuracy
    """
    attack_model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = attack_model(X_batch)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(y_batch.numpy())

    auc = roc_auc_score(all_true, all_preds)
    acc = accuracy_score(all_true, [1 if p > 0.5 else 0 for p in all_preds])

    return auc, acc
