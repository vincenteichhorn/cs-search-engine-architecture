import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ListNet(nn.Module):
    def __init__(self, in_features, dropout=0.1, means=None, stds=None):
        super(ListNet, self).__init__()
        self.means = torch.tensor(means, dtype=torch.float32) if means is not None else None
        self.stds = torch.tensor(stds, dtype=torch.float32) if stds is not None else None
        assert dropout >= 0.0 and dropout < 1.0, "Dropout must be in [0.0, 1.0)"
        assert in_features > 0, "in_features must be positive."
        assert isinstance(in_features, int), "in_features must be an integer."
        assert isinstance(dropout, float), "dropout must be a float."
        assert in_features == self.means.shape[0], "in_features must match means length."
        assert in_features == self.stds.shape[0], "in_features must match stds length."
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X):
        # X has shape (batch_size, num_docs, num_features)
        # normalize features if means and stds are provided
        if self.means is not None and self.stds is not None:
            X = (X - self.means) / self.stds
        scores = self.net(X).squeeze(-1)  # shape (batch_size, num_docs)
        return scores


class CrossEntropyRankLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyRankLoss, self).__init__()

    def forward(self, predictions, relevances):
        log_pred_probs = F.log_softmax(predictions, dim=1)
        target_probs = F.softmax(relevances, dim=1)

        return -(target_probs * log_pred_probs).sum(dim=1).mean()
