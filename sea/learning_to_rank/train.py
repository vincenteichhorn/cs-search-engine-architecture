import pandas as pd
from torch.utils.data import DataLoader
from sea.learning_to_rank.dataset import RankingDataset, collate_fn

DATASET_PATH = "./data/dataset.csv"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 16


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

    for batch in train_data_loader:
        features, labels = batch
        print("Features shape:", features.shape)
        print(features)
        print("Labels shape:", labels.shape)
        print(labels)
        break
