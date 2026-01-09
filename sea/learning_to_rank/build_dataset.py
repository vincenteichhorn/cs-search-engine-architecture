import json
import pandas as pd
from tqdm import tqdm
from sea.corpus import Corpus, py_string_processor
from sea.tokenizer import Tokenizer
from sea.learning_to_rank.feature_mapping import build_dataset

INDEX_PATH = "./data/indices/all"
MAPPING_PATH = "./data/id_mapping.csv"
DATASET_PATH = "./data/dataset_20.csv"
QUERIES_PATH = "./data/msmarco-doctrain-queries.tsv"
TOP100_PATH = "./data/msmarco-doctrain-top100.tsv"
QREL_PATH = "./data/msmarco-doctrain-qrels.tsv"
TOP_N = 20


def is_numeric(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def build_id_mapping():
    corpus = Corpus(INDEX_PATH, "")

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        oid = -1
        while True:
            try:
                oid += 1
                if oid >= 3_213_835:
                    break
                doc_str = corpus.py_get(oid, py_string_processor)
                cid = doc_str.split("\t")[0]
                print(f"Processed {oid} documents", end="\r")
                if cid[0] != "D" or not is_numeric(cid[1:]):
                    continue
                f.write(f"{cid},{oid}\n")
            except Exception:
                pass


if __name__ == "__main__":

    # build_id_mapping()

    id_mapping_df = pd.read_csv(MAPPING_PATH, header=None, names=["doc_id", "our_id"])

    queries_df = pd.read_csv(QUERIES_PATH, sep="\t", header=None, names=["query_id", "query_text"])

    qrel_df = pd.read_csv(
        QREL_PATH, sep=" ", header=None, names=["query_id", "unused", "doc_id", "label"]
    )
    pos_df = pd.merge(queries_df, qrel_df, how="inner", on="query_id")
    pos_df = pd.merge(
        pos_df,
        id_mapping_df,
        how="inner",
        on="doc_id",
    )
    pos_df["rank"] = 1

    top100_df = pd.read_csv(
        TOP100_PATH,
        sep=" ",
        header=None,
        names=["query_id", "dummy", "doc_id", "rank", "score", "dummy2"],
    )
    top100_df = top100_df[top100_df["rank"] <= int(1.5 * TOP_N)]

    df = pd.merge(
        top100_df,
        id_mapping_df,
        how="inner",
        on="doc_id",
    )
    df = pd.merge(
        df,
        queries_df,
        how="inner",
        on="query_id",
    )

    # recompute the rank colum so that each query's documents are ranked from 1 to N, take only the top N (10)
    df = df.sort_values(by=["query_id", "score"], ascending=[True, False]).reset_index(drop=True)
    df["rank"] = df.groupby("query_id").cumcount() + 1
    df = df[df["rank"] < TOP_N]
    df["rank"] += 1

    # drop columns we don't need
    df = df[["query_id", "query_text", "our_id", "rank"]]
    # add positive samples
    df = pd.concat([df, pos_df[["query_id", "query_text", "our_id", "rank"]]], ignore_index=True)
    # sort by query_id and rank
    df = df.sort_values(by=["query_id", "rank"]).reset_index(drop=True)

    # df.head(100).to_csv("./data/_test.csv", index=False)
    # df = pd.read_csv("./data/_test.csv")
    print(f"Total queries in dataset: {df['query_id'].nunique()}.")

    with open(f"{INDEX_PATH}/meta.json", "r", encoding="utf-8") as f:
        index_meta = json.load(f)
    num_docs = index_meta["num_documents"]
    bm25_k = index_meta["bm25_k"]
    bm25_bs = index_meta["bm25_bs"]
    average_field_lengths = index_meta["avg_field_lengths"]
    document_frequencies = index_meta["term_document_frequencies"]
    print("Index metadata loaded.")

    build_dataset(
        INDEX_PATH,
        DATASET_PATH,
        df,
        num_docs,
        bm25_k,
        bm25_bs,
        average_field_lengths,
        document_frequencies,
    )
