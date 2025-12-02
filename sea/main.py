import os
import time
from sea.tokenizer import Tokenizer
from sea.engine import Engine
from sea.indexer import Indexer
from sea.query import Query
from sea.util.load import load_documents
import json


class SEAConfig:
    INDEX_PATH = "./data/indices/all"
    DOCUMENTS_PATH = "./data/msmarco-docs.tsv"
    MAX_DOCUMENTS = 3_213_835  # 3_213_835
    PARTITION_SIZE = 10_000
    DOCUMENTS_DATA_FILE_NAME = "documents.dat"
    DOCUMENTS_INDEX_FILE_NAME = "documents.idx"
    POSTINGS_DATA_FILE_NAME = "postings.dat"
    POSTINGS_INDEX_FILE_NAME = "postings.idx"
    PARTITION_PREFIX = "part"
    TIER_PREFIX = "tier"
    REINDEX_DOCUMENTS = True
    NUM_FIELDS = 2
    FIELD_BOOSTS = [1.0, 0.5]
    NUM_TIERS = 4
    TIER_SCORE_THRESHOLDS = [20, 10, 5, 0]
    BM25_K = 1.5
    BM25_B_VALUES = [0.75, 0.75]
    SPELLING_FREQUENCY_THRESHOLD = 100
    SNIPPET_RADIUS = 150
    IN_PHRASE_CHARACTER_DISTANCE = 15


def bold_string(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def main():

    tokenizer = Tokenizer()

    indexer = Indexer(SEAConfig)
    # indexer.add_documents(
    #     load_documents(SEAConfig.DOCUMENTS_PATH, tokenizer, SEAConfig.MAX_DOCUMENTS)
    # )

    with open(os.path.join(SEAConfig.INDEX_PATH, "index_meta.json"), "r") as f:
        meta = json.load(f)
    indexer.num_total_documents = meta["num_total_documents"]
    indexer.num_total_postings = meta["num_total_postings"]
    for i in range(SEAConfig.NUM_FIELDS):
        indexer.summed_field_lengths[i] = meta["summed_field_lengths"][i]
    indexer.global_doc_freqs = meta["global_doc_freqs"]
    indexer.merge_partitions()

    engine = Engine(SEAConfig)
    while True:
        print("\n")
        query_text = input("Enter your search query: ")
        try:
            query = Query(query_text, tokenizer)
        except Exception as e:
            print(f"Error parsing query: {e}")
            continue
        start = time.time()
        results = engine.search(query, limit=10)
        end = time.time()
        print(bold_string(f"Search Results:"))
        print(f"Found {len(results)} results in {(end - start)*1000:.4f} milliseconds:")
        for score, doc, snippet in results:
            print("-" * os.get_terminal_size().columns)
            print(
                f"{bold_string('Title')}: {doc.title}",
                f"{bold_string('URL')}: {doc.url}",
                f"{bold_string('Score')}: {score:.4f}",
                sep="\n",
            )
            print(f"{bold_string('Snippet')}: {snippet}")


if __name__ == "__main__":
    main()
