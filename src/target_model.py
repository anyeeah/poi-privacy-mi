import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

class POIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class POITransformer(nn.Module):
    """
    Transformer-based Next POI Prediction Model.
    
    Input : sequence of POI categories (window_size,)
    Output: next POI category probabilities (num_poi,)
    """
    def __init__(self, num_poi=86, embed_dim=64, 
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_poi, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_poi)

    def forward(self, x):
        x = self.embedding(x)       # (batch, seq_len, embed_dim)
        x = self.transformer(x)     # (batch, seq_len, embed_dim)
        x = x[:, -1, :]             # 마지막 토큰
        x = self.classifier(x)      # (batch, num_poi)
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += y_batch.size(0)
        correct1 += (output.argmax(dim=1) == y_batch).sum().item()
        pred5 = output.topk(5, dim=1).indices
        correct5 += (pred5 == y_batch.unsqueeze(1)).any(dim=1).sum().item()

    return total_loss / len(loader), correct1 / total, correct5 / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)

            total_loss += loss.item()
            total += y_batch.size(0)
            correct1 += (output.argmax(dim=1) == y_batch).sum().item()
            pred5 = output.topk(5, dim=1).indices
            correct5 += (pred5 == y_batch.unsqueeze(1)).any(dim=1).sum().item()

    return total_loss / len(loader), correct1 / total, correct5 / total
