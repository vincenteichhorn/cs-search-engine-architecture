import os
import pandas as pd
import torch
import json
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from sea.learning_to_rank.dataset import RankingDataset, collate_fn
from sea.learning_to_rank.model import ListNet

import wandb  # type: ignore

DATASET_PATH = "./data/dataset_10.csv"
SAVE_DIR = "./models/"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 32
EPOCHS = 1


def ndcg_at_k(predictions, ranks, k):
    """
    NDCG@k for rank labels (1 = best).
    """
    batch_size, num_docs = predictions.shape
    ndcg_scores = []

    for i in range(batch_size):
        # Convert ranks → relevance (higher is better)
        max_rank = ranks[i].max()
        # Determine relevance
        if ranks[i].max() > 1:  # ordinal ranks: smaller = better
            max_rank = ranks[i].max()
            relevance = max_rank - ranks[i] + 1  # higher is better
        else:  # binary relevance (0/1)
            relevance = ranks[i].float()

        # Predicted ranking
        _, pred_order = torch.sort(predictions[i], descending=True)
        pred_order = pred_order[:k]

        # DCG
        dcg = 0.0
        for rank_idx, doc_idx in enumerate(pred_order):
            rel = relevance[doc_idx]
            dcg += (2**rel - 1) / math.log2(rank_idx + 2)

        # Ideal DCG
        ideal_relevance, _ = torch.sort(relevance, descending=True)
        ideal_relevance = ideal_relevance[:k]

        idcg = 0.0
        for rank_idx, rel in enumerate(ideal_relevance):
            idcg += (2**rel - 1) / math.log2(rank_idx + 2)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores)


def mrr_at_k(predictions, ranks, k):
    """
    MRR@k for rank labels (1 = best).
    """
    batch_size = predictions.size(0)
    mrr_scores = []

    for i in range(batch_size):
        _, pred_order = torch.sort(predictions[i], descending=True)

        rr = 0.0
        for rank_idx in range(min(k, len(pred_order))):
            doc_idx = pred_order[rank_idx]
            if ranks[i][doc_idx] == 1:  # best document
                rr = 1.0 / (rank_idx + 1)
                break

        mrr_scores.append(rr)

    return sum(mrr_scores) / len(mrr_scores)


if __name__ == "__main__":

    dataset_df = pd.read_csv(DATASET_PATH)
    print(dataset_df.head())

    unique_queries = dataset_df["query_id"].unique()
    documents_per_query = dataset_df.groupby("query_id").size().values[0]
    num_queries = len(unique_queries)
    train_end = int(num_queries * TRAIN_RATIO) * documents_per_query
    valid_end = int(num_queries * (TRAIN_RATIO + VALID_RATIO)) * documents_per_query

    train_df = dataset_df.iloc[:train_end]
    valid_df = dataset_df.iloc[train_end:valid_end]
    test_df = dataset_df.iloc[valid_end:]

    train_data_loader = DataLoader(
        RankingDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    valid_data_loader = DataLoader(
        RankingDataset(valid_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_data_loader = DataLoader(
        RankingDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = ListNet(in_features=train_df.shape[1] - 3)
    config = {
        "in_features": train_df.shape[1] - 3,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    wandb_run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "sea-ltr"),
        name=os.getenv("WANDB_RUN_NAME", "listnet"),
        config={
            "dataset_path": DATASET_PATH,
            "train_ratio": TRAIN_RATIO,
            "valid_ratio": VALID_RATIO,
            "test_ratio": TEST_RATIO,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": 0.0001,
            "model": "ListNet",
        },
    )
    pbar = tqdm(range(EPOCHS * len(train_data_loader)), desc="Training", unit="step", position=0)
    i = 0

    for epoch in range(EPOCHS):
        for batch in train_data_loader:

            if i % 1000 == 0:

                ndcg_sum = 0.0
                mrr_sum = 0.0
                loss_sum = 0.0
                count = 0
                model.eval()
                with torch.no_grad():

                    for val_batch in tqdm(
                        valid_data_loader, desc="Validation", leave=False, position=1
                    ):
                        val_features, val_labels = val_batch
                        val_predictions = model(val_features)
                        val_loss = model.loss(val_predictions, val_labels)
                        loss_sum += val_loss.item() * val_features.size(0)
                        ndcg_sum += ndcg_at_k(
                            val_predictions, val_labels, k=10
                        ) * val_features.size(0)
                        mrr_sum += mrr_at_k(val_predictions, val_labels, k=10) * val_features.size(
                            0
                        )
                        count += val_features.size(0)

                    avg_val_loss = loss_sum / count if count > 0 else 0.0
                    avg_val_ndcg = ndcg_sum / count if count > 0 else 0.0
                    avg_val_mrr = mrr_sum / count if count > 0 else 0.0

                    pbar.write(
                        f"Validation Loss: {avg_val_loss:.4f}, Validation NDCG@10: {avg_val_ndcg:.4f}, MRR@10: {avg_val_mrr:.4f}"
                    )

                    # Log validation metrics
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "valid/loss": float(avg_val_loss),
                                "valid/ndcg@10": float(avg_val_ndcg),
                                "valid/mrr@10": float(avg_val_mrr),
                                "epoch": epoch + 1,
                                "step": i,
                            }
                        )
                # Save model checkpoint
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                latest_model_path = os.path.join(SAVE_DIR, "listnet_latest.pth")
                torch.save(model.state_dict(), latest_model_path)
                config_path = os.path.join(SAVE_DIR, "listnet_config.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                pbar.write(f"Saved latest model to {latest_model_path}")

                model.train()

            i += 1
            features, labels = batch
            optimizer.zero_grad()
            predictions = model(features)
            loss = model.loss(predictions, labels)
            loss.backward()
            optimizer.step()
            pbar.update(1)

            predictions[predictions != 1.0] = 0.0  # Binarize predictions for metric calculation
            ndcg = ndcg_at_k(predictions, labels, k=10)
            mrr = mrr_at_k(predictions, labels, k=10)
            pbar.set_postfix(
                {
                    "epoch": epoch + 1,
                    "loss": f"{loss.item():.4f}",
                    "NDCG@10": f"{ndcg:.4f}",
                    "MRR@10": f"{mrr:.4f}",
                }
            )

            if wandb_run is not None:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/ndcg@10": float(ndcg),
                        "train/mrr@10": float(mrr),
                        "epoch": epoch + 1,
                        "step": i,
                    }
                )
        break

    # Finish WandB run if started
    if wandb_run is not None:
        wandb.finish()
