import os
import pandas as pd
import numpy as np
import torch
import json
import wandb
import math
import math
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sea.learning_to_rank.dataset import RankingDataset, collate_fn
from sea.learning_to_rank.model import (
    CrossEntropyRankLoss,
    ListNet,
)


DATASET_PATH = "./data/dataset_7_top50.csv"
NUM_DOCS_PER_QUERY = 50
SAVE_DIR = "./data/models/"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def ndcg_at_k_random_baseline(N, k):
    return (((2 ** (N + 1) - N - 2) / N) * sum(1 / math.log2(i + 1) for i in range(1, k + 1))) / sum(
        (2 ** (N - i + 1) - 1) / math.log2(i + 1) for i in range(1, k + 1)
    )


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


def make_dataloaders(train_df, valid_df, test_df, num_docs_per_query, batch_size):
    train_loader = DataLoader(
        RankingDataset(train_df, num_docs_per_query),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        RankingDataset(valid_df, num_docs_per_query),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        RankingDataset(test_df, num_docs_per_query),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader, test_loader


def save_model(model, config, save_dir=SAVE_DIR, filename="listnet_latest.pth"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), model_path)
    config_path = os.path.join(save_dir, "listnet_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)


def evaluate(model, dataloader, criterion, k=10):
    model.eval()
    all_predictions = []
    all_relevances = []
    all_losses = []

    with torch.no_grad():
        for features, ranks in tqdm(dataloader, desc="Evaluating", unit="batch", position=1, leave=False):
            outputs = model(features)
            loss = criterion(outputs, ranks)
            all_losses.append(loss.item())
            all_predictions.append(outputs)
            all_relevances.append(ranks)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_relevances = torch.cat(all_relevances, dim=0)

    ndcg_score = ndcg_at_k(all_predictions, all_relevances, k)
    mrr_score = mrr_at_k(all_predictions, all_relevances, k)
    avg_loss = sum(all_losses) / len(all_losses)

    return ndcg_score, mrr_score, avg_loss


def train(
    model,
    config,
    criterion,
    train_dataloader,
    optimizer,
    wandb_run=None,
    validate_dataloader=None,
    validate_every=100,
    epoch=1,
):
    model.train()
    pbar = tqdm(
        desc=f"Training (Epoch {epoch})",
        unit="batch",
        total=len(train_dataloader),
        position=0,
        leave=False,
    )
    best_val_loss = float("inf")
    for batch_idx, (features, ranks) in enumerate(train_dataloader):
        pbar.update(1)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, ranks)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            {
                "train/loss": f"{loss.item():.4f}",
            }
        )
        if wandb_run:
            wandb_run.log(
                {
                    "step": batch_idx + epoch * len(train_dataloader),
                    "train/loss": loss.item(),
                }
            )
        if validate_dataloader and ((batch_idx + 1) % validate_every == 0 or batch_idx == 0):
            ndcg_score, mrr_score, val_loss = evaluate(model, validate_dataloader, criterion, k=10)
            pbar.write(f"val/loss: {val_loss:.4f}, val/ndcg@10: {ndcg_score:.4f}, val/mrr@10: {mrr_score:.4f}")
            if wandb_run:
                wandb_run.log(
                    {
                        "step": (batch_idx + 1) + epoch * len(train_dataloader),
                        "val/loss": val_loss,
                        "val/ndcg@10": ndcg_score,
                        "val/mrr@10": mrr_score,
                    }
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(
                    model,
                    config,
                    save_dir=SAVE_DIR,
                )
                pbar.write(f"Model saved with val/loss: {best_val_loss:.4f}")

    pbar.close()


if __name__ == "__main__":
    dataset_df = pd.read_csv(DATASET_PATH)

    num_queries = 64000
    dataset_df = dataset_df[dataset_df["query_id"].isin(dataset_df["query_id"].unique()[:num_queries])].reset_index(drop=True)

    print(dataset_df.head())

    batch_size = 64
    train_df, valid_df, test_df = split_dataset(dataset_df)
    train_dataloader, validate_dataloader, test_dataloader = make_dataloaders(
        train_df, valid_df, test_df, num_docs_per_query=NUM_DOCS_PER_QUERY, batch_size=batch_size
    )

    in_features = train_df.shape[1] - 2  # exclude query_id and rank
    means, stds = train_dataloader.dataset.means_and_stds()
    config = {
        "in_features": in_features,
        "dropout": 0.2,
        "means": means,
        "stds": stds,
        "learning_rate": 1e-5,
        "temperature": 0.5,
        "batch_size": batch_size,
        "epochs": 3,
    }
    model = ListNet(**config)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    criterion = CrossEntropyRankLoss(temperature=config["temperature"])

    wandb_run = wandb.init(project="sea-ltr", config=config)
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        train(
            model,
            config,
            criterion,
            train_dataloader,
            optimizer,
            wandb_run=wandb_run,
            validate_dataloader=validate_dataloader,
            validate_every=100,
            epoch=epoch + 1,
        )
