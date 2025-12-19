import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNet(nn.Module):
    def __init__(self, in_features):
        super(ListNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, X):
        # X has shape (batch_size, num_docs, num_features)
        return self.net(X).squeeze(-1)  # shape (batch_size, num_docs)
