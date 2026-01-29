from torch.utils.data import Dataset
import torch


class RankingDataset(Dataset):
    def __init__(self, dataframe, num_docs_per_query=10):
        self.dataframe = dataframe
        # drop all queries that do not have 10 documents
        counts = self.dataframe["query_id"].value_counts()
        valid_query_ids = counts[counts == num_docs_per_query].index
        self.dataframe = self.dataframe[self.dataframe["query_id"].isin(valid_query_ids)].reset_index(drop=True)
        self.num_queries = self.dataframe["query_id"].nunique()
        query_id_mapping = {old_id: new_id for new_id, old_id in enumerate(self.dataframe["query_id"].unique())}
        self.dataframe.loc[:, "query_id"] = self.dataframe["query_id"].map(query_id_mapping)

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
        sub_df = self.dataframe[self.dataframe["query_id"] == idx]
        # shuffle the documents for this query
        sub_df = sub_df.sample(frac=1).reset_index(drop=True)
        features = torch.tensor(sub_df.drop(columns=["query_id", "rank"]).values, dtype=torch.float32)
        max_rank = sub_df["rank"].max()
        relevances = torch.tensor(max_rank - sub_df["rank"].values + 1, dtype=torch.float32)

        return features, relevances


def collate_fn(batch):
    features_batch = torch.stack([item[0] for item in batch])  # shape (batch_size, num_docs, num_features)
    relevances_batch = torch.stack([item[1] for item in batch])  # shape (batch_size, num_docs)
    return features_batch, relevances_batch
