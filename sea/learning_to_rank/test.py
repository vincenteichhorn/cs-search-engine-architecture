import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sea.learning_to_rank.build_dataset import DATASET_PATH
from sea.learning_to_rank.model import ListNet
from sea.learning_to_rank.train import BATCH_SIZE, NUM_DOCS_PER_QUERY, TRAIN_RATIO, VALID_RATIO, make_dataloaders, mrr_at_k, ndcg_at_k, split_dataset


if __name__ == "__main__":

    MODEL_NAME = "all"
    MODEL_PATH = "./data/models/"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_df = pd.read_csv(DATASET_PATH)
    # dataset_df = dataset_df[dataset_df["query_id"].isin(dataset_df["query_id"].unique()[:NUM_QUERIES])].reset_index(drop=True)

    train_df, valid_df, test_df = split_dataset(dataset_df, TRAIN_RATIO, VALID_RATIO)

    train_dataloader, validate_dataloader, test_dataloader = make_dataloaders(
        train_df, valid_df, test_df, num_docs_per_query=NUM_DOCS_PER_QUERY, batch_size=BATCH_SIZE, device=device
    )

    with open(os.path.join(MODEL_PATH, f"{MODEL_NAME}.json"), "r") as f:
        config = json.load(f)
    ltr_model = ListNet(**config)
    ltr_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{MODEL_NAME}.pth"), map_location=device))
    ltr_model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    all_predictions = []
    all_relevances = []
    all_losses = []
    with torch.no_grad():
        test_pbar = tqdm(desc="Test Batches", total=len(test_dataloader), leave=False)
        for test_batch in test_dataloader:
            test_pbar.update(1)
            test_features, test_labels = test_batch
            test_features, test_labels = test_features.to(device), test_labels.to(device)
            test_outputs = ltr_model(test_features)
            loss = criterion(test_outputs, test_labels)
            all_predictions.append(test_outputs)
            all_relevances.append(test_labels)
            all_losses.append(loss.item())
        test_pbar.close()
    all_predictions = torch.cat(all_predictions, dim=0)
    all_relevances = torch.cat(all_relevances, dim=0)
    avg_val_loss = np.mean(all_losses)
    ndcg_10 = ndcg_at_k(all_predictions, all_relevances, k=10)
    mrr_10 = mrr_at_k(all_predictions, all_relevances, k=10)

    print(f"Test Loss: {avg_val_loss:.4f}, NDCG@10: {ndcg_10:.4f}, MRR@10: {mrr_10:.4f}")
