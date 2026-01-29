from torch.utils.data import Dataset
import torch
from tqdm import tqdm


class RankingDataset(Dataset):
    def __init__(self, dataframe, num_docs_per_query=10, device="cuda"):
        self.dataframe = dataframe
        # Drop queries that do not have exactly `num_docs_per_query` documents
        counts = self.dataframe["query_id"].value_counts()
        valid_query_ids = counts[counts == num_docs_per_query].index
        self.dataframe = self.dataframe[self.dataframe["query_id"].isin(valid_query_ids)].reset_index(drop=True)
        self.num_queries = self.dataframe["query_id"].nunique()

        # Map query IDs to a continuous range
        query_id_mapping = {old_id: new_id for new_id, old_id in enumerate(self.dataframe["query_id"].unique())}
        self.dataframe.loc[:, "query_id"] = self.dataframe["query_id"].map(query_id_mapping)

        # Precompute and store features and relevances as tensors
        self.features_list = []
        self.relevances_list = []

        for query_id, sub_df in tqdm(self.dataframe.groupby("query_id"), desc="Processing queries"):
            features = torch.tensor(sub_df.drop(columns=["query_id", "rank"]).values, dtype=torch.float32)
            max_rank = sub_df["rank"].max()
            relevances = torch.tensor(max_rank - sub_df["rank"].values + 1, dtype=torch.float32)

            self.features_list.append(features)
            self.relevances_list.append(relevances)

        # Move all tensors to the specified device
        self.features_list = [features.to(device) for features in tqdm(self.features_list, desc="Moving features to device")]
        self.relevances_list = [relevances.to(device) for relevances in tqdm(self.relevances_list, desc="Moving relevances to device")]

    def means_and_stds(self):
        feature_columns = self.dataframe.drop(columns=["query_id", "rank"]).columns
        features = self.dataframe[feature_columns].values
        means = list(features.mean(axis=0))
        stds = list(features.std(axis=0))
        return means, stds

    def __len__(self):
        return self.num_queries

    def __getitem__(self, idx):
        assert idx < self.num_queries, "Index out of range."
        return self.features_list[idx], self.relevances_list[idx]


def collate_fn(batch):
    features_batch = torch.stack([item[0] for item in batch])  # shape (batch_size, num_docs, num_features)
    relevances_batch = torch.stack([item[1] for item in batch])  # shape (batch_size, num_docs)
    return features_batch, relevances_batch
