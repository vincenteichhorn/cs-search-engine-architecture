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
        scores = self.net(X).squeeze(-1)  # shape (batch_size, num_docs)
        return scores

    def loss(self, predictions, ranks):
        # Convert ranks to relevance
        max_rank = ranks.max(dim=1, keepdim=True).values
        relevance = max_rank - ranks + 1  # higher = better

        log_pred_probs = F.log_softmax(predictions, dim=1)
        target_probs = F.softmax(relevance, dim=1)

        return -(target_probs * log_pred_probs).sum(dim=1).mean()
