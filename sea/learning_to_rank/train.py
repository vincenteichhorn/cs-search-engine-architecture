import argparse
import os
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import json
import wandb
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW
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


def dcgs_at_k(predictions: torch.Tensor, relevances: torch.Tensor, k: int) -> torch.Tensor:
    """
    Vectorized GPU implementation of DCG@k.
    """
    k = min(k, predictions.shape[1])
    _, indices = torch.sort(predictions, descending=True, dim=1)
    top_k_indices = indices[:, :k]

    gathered_relevances = torch.gather(relevances, 1, top_k_indices)
    ranks = torch.arange(1, k + 1, device=predictions.device, dtype=torch.float32)
    denominators = torch.log2(ranks + 1.0)

    gains = 2**gathered_relevances - 1

    return torch.sum(gains / denominators, dim=1)


def ndcg_at_k(predictions: torch.Tensor, relevances: torch.Tensor, k: int) -> float:
    """
    Vectorized GPU implementation of NDCG@k.
    """
    dcg = dcgs_at_k(predictions, relevances, k)
    idcg = dcgs_at_k(relevances, relevances, k)
    ndcg = dcg / (idcg + 1e-8)
    return ndcg.mean().item()


def mrr_at_k(predictions: torch.Tensor, relevances: torch.Tensor, k: int) -> float:
    """
    MRR@k for rank labels (1 = best).
    predictions: Tensor of shape (batch_size, num_docs) with predicted relevance scores.
    relevances: Tensor of shape (batch_size, num_docs) with ground truth relevances (higher = better).
    k: int, the cutoff rank.
    """
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    top_k_relevances = torch.gather(relevances, 1, top_k_indices)
    max_rel, _ = torch.max(relevances, dim=1, keepdim=True)
    hits = top_k_relevances == max_rel
    hit_ranks = torch.argmax(hits.to(torch.int), dim=1)
    found_mask = hits.any(dim=1)
    rr = torch.where(found_mask, 1.0 / (hit_ranks.to(torch.float) + 1.0), torch.tensor(0.0, device=predictions.device))

    return rr.mean().item()


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
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, num_docs_per_query: int, batch_size: int, device="cuda"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        RankingDataset(train_df, num_docs_per_query, device=device),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        # num_workers=8,
    )
    valid_loader = DataLoader(
        RankingDataset(valid_df, num_docs_per_query, device=device),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        # num_workers=8,
    )
    test_loader = DataLoader(
        RankingDataset(test_df, num_docs_per_query, device=device),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        # num_workers=8,
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


DATASET_PATH = "./data/dataset_8_top50.csv"
SAVE_DIR = "./data/models/"
NUM_QUERIES = 10_000  # Limit number of queries for faster training during testing
NUM_DOCS_PER_QUERY = 50
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 256
DROPOUT = 0.1
LEARNING_RATE = 5e-4
EPOCHS = 3
LOSS_FN_TEMPERATURE = 0.7
VALIDATE_EVERY_N_STEPS = 100


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    argparser.add_argument("--temperature", type=float, default=LOSS_FN_TEMPERATURE)
    args = argparser.parse_args()
    LEARNING_RATE = args.learning_rate
    LOSS_FN_TEMPERATURE = args.temperature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_df = pd.read_csv(DATASET_PATH)
    # dataset_df = dataset_df[dataset_df["query_id"].isin(dataset_df["query_id"].unique()[:NUM_QUERIES])].reset_index(drop=True)

    train_df, valid_df, test_df = split_dataset(dataset_df, TRAIN_RATIO, VALID_RATIO)

    train_dataloader, validate_dataloader, test_dataloader = make_dataloaders(
        train_df, valid_df, test_df, num_docs_per_query=NUM_DOCS_PER_QUERY, batch_size=BATCH_SIZE, device=device
    )

    means, stds = train_dataloader.dataset.means_and_stds()

    learning_rates = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]
    dropouts = [0.0, 0.1, 0.2, 0.3]

    # iter all hyperparameter combinations
    combinations = [(lr, temp, drop) for lr in learning_rates for temp in temperatures for drop in dropouts]
    for i, (lr, temp, drop) in enumerate(combinations, 1):
        print(f"Training with learning rate: {lr}, temperature: {temp}, dropout: {drop}, config {i}/{len(combinations)}")
        LEARNING_RATE = lr
        LOSS_FN_TEMPERATURE = temp
        DROPOUT = drop

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
        model = ListNet(in_features=config["in_features"], dropout=config["dropout"], means=config["means"], stds=config["stds"]).to(device)
        criterion = CrossEntropyRankLoss(temperature=config["temperature"]).to(device)
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

        wandb.login()
        wandb_run = wandb.init(project="sea-ltr", config=config)
        model_name = wandb_run.name

        pbar = tqdm(desc="Training Batches", total=len(train_dataloader) * EPOCHS, position=0)

        global_step = 0
        best_val_loss = float("inf")
        for epoch in range(EPOCHS):
            for batch in train_dataloader:
                pbar.update(1)
                global_step += 1
                features, labels = batch
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                actual_lr = optimizer.param_groups[0]["lr"]

                pbar.set_postfix({"epoch": epoch + 1, "loss": loss.item()})
                wandb_run.log(
                    {
                        "step": global_step,
                        "train/loss": loss.item(),
                        "learning_rate": actual_lr,
                    }
                )

                if global_step % VALIDATE_EVERY_N_STEPS == 0 or global_step == 1:
                    model.eval()
                    all_predictions = []
                    all_relevances = []
                    all_losses = []
                    with torch.no_grad():
                        val_pbar = tqdm(desc="Validation Batches", total=len(validate_dataloader), position=1, leave=False)
                        for val_batch in validate_dataloader:
                            val_pbar.update(1)
                            val_features, val_labels = val_batch
                            val_features, val_labels = val_features.to(device), val_labels.to(device)
                            val_outputs = model(val_features)
                            loss = criterion(val_outputs, val_labels)
                            all_predictions.append(val_outputs)
                            all_relevances.append(val_labels)
                            all_losses.append(loss.item())
                        val_pbar.close()
                    all_predictions = torch.cat(all_predictions, dim=0)
                    all_relevances = torch.cat(all_relevances, dim=0)
                    avg_val_loss = np.mean(all_losses)
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_model(model, config, SAVE_DIR, model_name)

                    ndcg_10 = ndcg_at_k(all_predictions, all_relevances, k=10)
                    mrr_10 = mrr_at_k(all_predictions, all_relevances, k=10)

                    pbar.write(f"Validation NDCG@10: {ndcg_10:.4f}/0.31, MRR@10: {mrr_10:.4f}/0.25, Loss: {avg_val_loss:.4f}")
                    wandb_run.log({"step": global_step, "valid/ndcg@10": ndcg_10, "valid/mrr@10": mrr_10, "valid/loss": avg_val_loss})
                    model.train()

        pbar.close()
        wandb_run.finish()
