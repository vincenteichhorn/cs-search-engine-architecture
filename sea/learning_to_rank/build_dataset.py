import pandas as pd
from tqdm import tqdm
from sea.corpus import Corpus, py_string_processor
from sea.engine import Engine

INDEX_PATH = "./data/indices/all"
MAPPING_PATH = "./data/id_mapping.csv"
DATASET_ID_PATH = "./data/dataset_ids.csv"
DATASET_PATH = "./data/dataset_7_top50.csv"
QUERIES_PATH = "./data/msmarco-doctrain-queries.tsv"
TOP100_PATH = "./data/msmarco-doctrain-top100.tsv"
QREL_PATH = "./data/msmarco-doctrain-qrels.tsv"
TOP_N = 50


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


def build_id_dataset():
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
    top100_df = top100_df[top100_df["rank"] <= TOP_N * 1.5].reset_index(drop=True)

    top100_df = pd.merge(
        top100_df,
        id_mapping_df,
        how="inner",
        on="doc_id",
    )
    top100_df = pd.merge(
        top100_df,
        queries_df,
        how="inner",
        on="query_id",
    )

    pos_df = pos_df[["query_id", "query_text", "doc_id", "our_id", "rank"]].set_index(
        ["query_id", "our_id"]
    )
    print(pos_df.head())
    print("Positive samples shape:", pos_df.shape, pos_df.columns)
    top100_df = top100_df[["query_id", "query_text", "doc_id", "our_id", "rank"]].set_index(
        ["query_id", "our_id"]
    )
    print(top100_df.head())
    print("Top100 samples shape:", top100_df.shape, top100_df.columns)

    # remove positive samples from top100
    top100_df = top100_df.drop(index=pos_df.index.intersection(top100_df.index))
    print("Top100 samples after removing positives shape:", top100_df.shape)

    pos_df = pos_df.reset_index(level=["query_id", "our_id"])
    top100_df = top100_df.reset_index(level=["query_id", "our_id"])

    df = pd.concat([pos_df, top100_df], ignore_index=True)
    print(df.columns)
    # sort by query_id and rank, the NaN value (of the positive samples) should be at the top
    df = df.sort_values(by=["query_id", "rank"]).reset_index(drop=True)
    df["rank"] = df.groupby("query_id").cumcount() + 1  # re-cumulate ranks
    # remove ranks greater than TOP_N
    df = df[df["rank"] <= TOP_N].reset_index(drop=True)
    print(df.head(TOP_N))
    # check for duplicates
    assert df.duplicated(subset=["query_id", "our_id"]).sum() == 0

    df.to_csv(DATASET_ID_PATH, index=False)


if __name__ == "__main__":

    # build_id_mapping()

    # build_id_dataset()

    dataset_df = pd.read_csv(DATASET_ID_PATH)

    engine = Engine(INDEX_PATH)

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        f.write(
            "query_id,bm25_title,bm25_body,title_length,body_length,ratio_query_in_title,ratio_query_in_body,first_occurrence_document,rank\n"
        )
        for query_id, group in tqdm(dataset_df.groupby("query_id")):
            query_text = group.iloc[0]["query_text"]
            doc_ids = group["our_id"].tolist()
            mat = engine.simulate_feature_matrix(doc_ids, query_text)
            num_features = mat.shape[1]
            num_docs = mat.shape[0]
            for i in range(num_docs):
                f.write(str(query_id) + ",")
                for j in range(num_features):
                    f.write(str(mat[i, j]))
                    if j < num_features - 1:
                        f.write(",")
                f.write("," + str(group.iloc[i]["rank"]))
                f.write("\n")
