from torch.utils.data import Dataset
import torch


class RankingDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.num_queries = dataframe["query_id"].nunique()
        query_id_mapping = {
            old_id: new_id for new_id, old_id in enumerate(dataframe["query_id"].unique())
        }
        self.dataframe["query_id"] = self.dataframe["query_id"].map(query_id_mapping)

    def __len__(self):
        return self.num_queries

    def __getitem__(self, idx):

        assert idx < self.num_queries, "Index out of range."
        sub_df = self.dataframe[self.dataframe["query_id"] == idx]
        features = torch.tensor(
            sub_df.drop(columns=["query_id", "doc_id"]).values, dtype=torch.float32
        )
        ranks = torch.arange(1, len(sub_df) + 1, dtype=torch.float32)

        return features, ranks


def collate_fn(batch):
    features_batch = torch.stack([item[0] for item in batch])
    ranks_batch = torch.stack([item[1] for item in batch])
    return features_batch, ranks_batch
