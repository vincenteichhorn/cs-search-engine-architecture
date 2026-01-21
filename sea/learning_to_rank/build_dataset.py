import os
from concurrent.futures import ProcessPoolExecutor, as_completed, FIRST_COMPLETED, wait

import pandas as pd
from tqdm import tqdm
from sea.corpus import Corpus, py_string_processor
from sea.engine import Engine

INDEX_PATH = "./data/indices/all"
MAPPING_PATH = "./data/id_mapping.csv"
DATASET_ID_PATH = "./data/dataset_7_ids.csv"
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


# Global engine per worker process
_ENGINE = None


def _init_worker(index_path: str):
    global _ENGINE
    _ENGINE = Engine(index_path)


def _process_query(task):
    query_id, query_text, doc_ids, ranks = task
    mat = _ENGINE.simulate_feature_matrix(doc_ids, query_text)
    num_features = mat.shape[1]
    lines = []
    for i in range(mat.shape[0]):
        parts = [str(query_id)]
        parts.extend(str(mat[i, j]) for j in range(num_features))
        parts.append(str(ranks[i]))
        lines.append(",".join(parts))
    return query_id, lines


if __name__ == "__main__":

    # build_id_mapping()

    # build_id_dataset()

    def _build_task_from_rows(qid, rows):
        query_text = rows[0].query_text
        doc_ids = [int(r.our_id) for r in rows]
        ranks = [int(r.rank) for r in rows]
        return int(qid), query_text, doc_ids, ranks

    def _iter_query_tasks(csv_path: str, chunksize: int = 200_000):
        current_qid = None
        buffer_rows = []
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            for row in chunk.itertuples(index=False):
                qid = row.query_id
                if current_qid is None:
                    current_qid = qid
                if qid != current_qid:
                    yield _build_task_from_rows(current_qid, buffer_rows)
                    buffer_rows = []
                    current_qid = qid
                buffer_rows.append(row)
        if buffer_rows:
            yield _build_task_from_rows(current_qid, buffer_rows)

    def _count_queries(csv_path: str, chunksize: int = 500_000) -> int:
        count = 0
        current_qid = None
        for chunk in pd.read_csv(csv_path, usecols=["query_id"], chunksize=chunksize):
            for qid in chunk["query_id"].values:
                if current_qid is None or qid != current_qid:
                    count += 1
                    current_qid = qid
        return count

    os.makedirs(os.path.dirname(DATASET_PATH) or ".", exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        f.write(
            "query_id,bm25_title,bm25_body,title_length,body_length,ratio_query_in_title,ratio_query_in_body,first_occurrence_document,rank\n"
        )

        max_workers = 3
        max_in_flight = max_workers * 2
        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_worker, initargs=(INDEX_PATH,)
        ) as executor:
            future_to_qid = {}

            # Submit tasks lazily while keeping a bounded number of futures in-flight
            submitted = 0
            completed = 0
            total_queries = _count_queries(DATASET_ID_PATH)
            with tqdm(total=total_queries) as pbar:
                for task in _iter_query_tasks(DATASET_ID_PATH):
                    # Backpressure when too many futures are pending
                    while len(future_to_qid) >= max_in_flight:
                        done, _ = wait(list(future_to_qid.keys()), return_when=FIRST_COMPLETED)
                        for fut in done:
                            _, lines = fut.result()
                            for line in lines:
                                f.write(line + "\n")
                            completed += 1
                            pbar.update(1)
                            del future_to_qid[fut]
                    fut = executor.submit(_process_query, task)
                    future_to_qid[fut] = task[0]
                    submitted += 1

                # Drain remaining futures
                for fut in as_completed(list(future_to_qid.keys())):
                    _, lines = fut.result()
                    for line in lines:
                        f.write(line + "\n")
                    completed += 1
                    pbar.update(1)
