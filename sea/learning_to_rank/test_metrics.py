import pandas as pd
from tqdm import tqdm
import torch
from sea.learning_to_rank.train import ndcg_at_k, mrr_at_k

TOP100_PATH = "./data/msmarco-doctrain-top100.tsv"
QREL_PATH = "./data/msmarco-doctrain-qrels.tsv"


if __name__ == "__main__":

    qrel_df = pd.read_csv(
        QREL_PATH, sep=" ", header=None, names=["query_id", "unused", "doc_id", "label"]
    )

    df = pd.read_csv(
        TOP100_PATH,
        sep=" ",
        header=None,
        names=["query_id", "dummy", "doc_id", "rank", "score", "dummy2"],
    )
    df = pd.merge(
        df,
        qrel_df[qrel_df["label"] > 0][["query_id", "doc_id", "label"]],
        how="left",
        on=["query_id", "doc_id"],
    )
    df["relevance"] = df["label"].apply(lambda x: 1 if pd.notna(x) else 0)
    df = df.drop(columns=["label"])

    print(df.head(10))
    ndcg_scores_sum = 0.0
    mrr_scores_sum = 0.0
    query_ids = df["query_id"].unique()
    pbar = tqdm(desc="Computing metrics", total=len(query_ids))
    count = 0
    for qid in query_ids:
        sub_df = df[df["query_id"] == qid]
        labels = torch.tensor([sub_df["relevance"].values])
        scores = torch.tensor([sub_df["score"].values])
        ndcg_score = float(ndcg_at_k(scores, labels, k=10))
        mrr_score = float(mrr_at_k(scores, labels, k=10))
        ndcg_scores_sum += ndcg_score
        mrr_scores_sum += mrr_score
        pbar.update(1)
        count += 1
        pbar.write(f"Query ID: {qid}, NDCG@10: {ndcg_score}, MRR@10: {mrr_score}")
        pbar.set_postfix({"Avg NDCG@10": ndcg_scores_sum / count, "MRR@10": mrr_scores_sum / count})
    pbar.close()
    avg_ndcg = ndcg_scores_sum / len(query_ids)
    avg_mrr = mrr_scores_sum / len(query_ids)
    print(f"Average NDCG@10: {avg_ndcg}")
    print(f"Average MRR@10: {avg_mrr}")
