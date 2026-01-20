import os
import pandas as pd
import numpy as np
import torch
import json
import wandb
import math
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from sea.learning_to_rank.dataset import RankingDataset, collate_fn
from sea.learning_to_rank.model import (
    CrossEntropyRankLoss,
    # LambdaRankLoss,
    ListNet,
)

rand_ndcg_at_k = lambda N, k: (
    ((2 ** (N + 1) - N - 2) / N) * sum(1 / math.log2(i + 1) for i in range(1, k + 1))
) / sum((2 ** (N - i + 1) - 1) / math.log2(i + 1) for i in range(1, k + 1))

DATASET_PATH = "./data/dataset_9_top20.csv"
NUM_DOCS_PER_QUERY = 20
SAVE_DIR = "./models/"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 32
EPOCHS = 5


def dcgs(predictions, relevances, k):
    """
    DCG@k for rank labels (1 = best).
    """
    batch_size = predictions.shape[0]
    dcg_scores = []

    for i in range(batch_size):
        _, pred_order = torch.sort(predictions[i], descending=True)
        dcg_score = 0.0

        for rank_idx in range(min(k, len(pred_order))):
            doc_idx = pred_order[rank_idx]
            rel_i = relevances[i][doc_idx]
            denom = math.log2(rank_idx + 2)  # rank_idx starts at 0
            dcg_score += (2**rel_i - 1) / denom

        dcg_scores.append(dcg_score)

    return np.array(dcg_scores)


def ndcg_at_k(predictions, relevances, k):
    """
    NDCG@k for rank labels (1 = best).
    """
    dcgs_scores = dcgs(predictions, relevances, k)
    ideal_dcgs_scores = dcgs(relevances, relevances, k)
    ndcg_scores = dcgs_scores / (ideal_dcgs_scores + 1e-8)
    return np.mean(ndcg_scores)


def mrr_at_k(predictions, relevances, k):
    """
    MRR@k for rank labels (1 = best).
    predictions: Tensor of shape (batch_size, num_docs) with predicted relevance scores.
    relevances: Tensor of shape (batch_size, num_docs) with ground truth relevances (higher = better).
    k: int, the cutoff rank.
    """
    batch_size = predictions.shape[0]
    mrr_scores = []

    for i in range(batch_size):
        _, pred_order = torch.sort(predictions[i], descending=True)
        rr = 0.0
        max_relevance = relevances[i].max()
        for rank_idx in range(min(k, len(pred_order))):
            doc_idx = pred_order[rank_idx]
            if relevances[i][doc_idx] == max_relevance:  # ground truth best relevance
                rr = 1.0 / (rank_idx + 1)
                break

        mrr_scores.append(rr)

    return np.array(mrr_scores).mean()


def split_dataset(df):
    unique_queries = df["query_id"].unique()
    documents_per_query = df.groupby("query_id").size().values[0]
    num_queries = len(unique_queries)
    train_end = int(num_queries * TRAIN_RATIO) * documents_per_query
    valid_end = int(num_queries * (TRAIN_RATIO + VALID_RATIO)) * documents_per_query

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]
    return train_df, valid_df, test_df


def make_dataloaders(train_df, valid_df, test_df, num_docs_per_query):
    train_loader = DataLoader(
        RankingDataset(train_df, num_docs_per_query),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        RankingDataset(valid_df, num_docs_per_query),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        RankingDataset(test_df, num_docs_per_query),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    dataset_df = pd.read_csv(DATASET_PATH)

    # # take only the first 1000 queries for faster experimentation
    # num_queries = 16000
    # dataset_df = dataset_df[
    #     dataset_df["query_id"].isin(dataset_df["query_id"].unique()[:num_queries])
    # ].reset_index(drop=True)

    print(dataset_df.head())

    train_df, valid_df, test_df = split_dataset(dataset_df)
    train_loader, valid_loader, test_loader = make_dataloaders(
        train_df, valid_df, test_df, num_docs_per_query=NUM_DOCS_PER_QUERY
    )

    in_features = train_df.shape[1] - 3
    means, stds = train_loader.dataset.means_and_stds()
    config = {"in_features": in_features, "dropout": 0.1, "means": means, "stds": stds}

    model = ListNet(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyRankLoss()
