import os
from typing import Tuple
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


def ndcg_at_k_random_baseline(N: int, k: int) -> float:
    """
    NDCG@k for random baseline given N documents.
    """
    return (((2 ** (N + 1) - N - 2) / N) * sum(1 / math.log2(i + 1) for i in range(1, k + 1))) / sum(
        (2 ** (N - i + 1) - 1) / math.log2(i + 1) for i in range(1, k + 1)
    )


def dcgs_at_k(predictions: torch.Tensor, relevances: torch.Tensor, k: int) -> np.ndarray:
    """
    DCG@k for predictions agains relevances (higher = better).
    """
    assert predictions.shape == relevances.shape, "Predictions and relevances must have the same shape."
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


def ndcg_at_k(predictions: torch.Tensor, relevances: torch.Tensor, k: int) -> float:
    """
    NDCG@k for predictions against relevances (higher = better).
    """
    assert predictions.shape == relevances.shape, "Predictions and relevances must have the same shape."
    dcgs_scores = dcgs_at_k(predictions, relevances, k)
    ideal_dcgs_scores = dcgs_at_k(relevances, relevances, k)
    ndcg_scores = dcgs_scores / (ideal_dcgs_scores + 1e-8)
    return np.mean(ndcg_scores)


def mrr_at_k(predictions: torch.Tensor, relevances: torch.Tensor, k: int) -> float:
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


def split_dataset(df: pd.DataFrame, train_ratio: float, valid_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_queries = df["query_id"].unique()
    documents_per_query = df.groupby("query_id").size().values[0]
    num_queries = len(unique_queries)
    train_end = int(num_queries * train_ratio) * documents_per_query
    valid_end = int(num_queries * (train_ratio + valid_ratio)) * documents_per_query

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]
    return train_df, valid_df, test_df


def make_dataloaders(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, num_docs_per_query: int, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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


def save_model(model, config, save_dir, model_name="listnet"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    config_path = os.path.join(save_dir, f"{model_name}.json")
    with open(config_path, "w") as f:
        json.dump(config, f)


DATASET_PATH = "./data/dataset_7_top50.csv"
SAVE_DIR = "./data/models/"
MODEL_NAME = "all"
NUM_QUERIES = 100_000  # Limit number of queries for faster training during testing
NUM_DOCS_PER_QUERY = 50
NUM_DOCS_PER_QUERY = 50
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 64
DROPOUT = 0.1
LEARNING_RATE = 5e-6
EPOCHS = 1
LOSS_FN_TEMPERATURE = 10
VALIDATE_EVERY_N_STEPS = 100


if __name__ == "__main__":
    dataset_df = pd.read_csv(DATASET_PATH)
    dataset_df = dataset_df[dataset_df["query_id"].isin(dataset_df["query_id"].unique()[:NUM_QUERIES])].reset_index(drop=True)

    train_df, valid_df, test_df = split_dataset(dataset_df, TRAIN_RATIO, VALID_RATIO)

    train_dataloader, validate_dataloader, test_dataloader = make_dataloaders(
        train_df, valid_df, test_df, num_docs_per_query=NUM_DOCS_PER_QUERY, batch_size=BATCH_SIZE
    )

    means, stds = train_dataloader.dataset.means_and_stds()
    config = {
        "in_features": train_df.shape[1] - 2,  # exclude query_id and rank
        "dropout": DROPOUT,
        "means": means,
        "stds": stds,
        "learning_rate": LEARNING_RATE,
        "temperature": LOSS_FN_TEMPERATURE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
    model = ListNet(in_features=config["in_features"], dropout=config["dropout"], means=config["means"], stds=config["stds"])
    criterion = CrossEntropyRankLoss(temperature=config["temperature"])
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)
    save_model(model, config, SAVE_DIR, MODEL_NAME)

    wandb.login()
    wandb_run = wandb.init(project="sea-ltr", config=config)

    pbar = tqdm(desc="Training Batches", total=len(train_dataloader) * EPOCHS, position=0)

    global_step = 0
    for epoch in range(EPOCHS):
        for batch in train_dataloader:
            pbar.update(1)
            global_step += 1
            features, labels = batch

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_ndcg_10 = ndcg_at_k(outputs.detach(), labels.detach(), k=10)
            train_mrr_10 = mrr_at_k(outputs.detach(), labels.detach(), k=10)

            lr_scheduler.step(loss.item())
            actual_lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix({"epoch": epoch + 1, "loss": loss.item()})
            wandb_run.log(
                {"step": global_step, "train/loss": loss.item(), "train/ndcg@10": train_ndcg_10, "train/mrr@10": train_mrr_10, "learning_rate": actual_lr}
            )

            if global_step % VALIDATE_EVERY_N_STEPS == 0:
                model.eval()
                all_predictions = []
                all_relevances = []
                all_losses = []
                with torch.no_grad():
                    val_pbar = tqdm(desc="Validation Batches", total=len(validate_dataloader), position=1, leave=False)
                    for val_batch in validate_dataloader:
                        val_pbar.update(1)
                        val_features, val_labels = val_batch
                        val_outputs = model(val_features)
                        loss = criterion(val_outputs, val_labels)
                        all_predictions.append(val_outputs)
                        all_relevances.append(val_labels)
                        all_losses.append(loss.item())
                    val_pbar.close()
                all_predictions = torch.cat(all_predictions, dim=0)
                all_relevances = torch.cat(all_relevances, dim=0)
                avg_val_loss = np.mean(all_losses)

                ndcg_10 = ndcg_at_k(all_predictions, all_relevances, k=10)
                mrr_10 = mrr_at_k(all_predictions, all_relevances, k=10)

                pbar.write(f"Validation NDCG@10: {ndcg_10:.4f}/0.31, MRR@10: {mrr_10:.4f}/0.25, Loss: {avg_val_loss:.4f}")
                wandb_run.log({"step": global_step, "valid/ndcg@10": ndcg_10, "valid/mrr@10": mrr_10, "valid/loss": avg_val_loss})
                model.train()

    pbar.close()
    save_model(model, config, SAVE_DIR, MODEL_NAME)
    wandb_run.finish()
