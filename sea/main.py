import time
from sea.tokenizer import Tokenizer
from sea.engine import Engine
from sea.indexer import Indexer
from sea.query import Query
from sea.util.load import load_documents


class SEAConfig:
    INDEX_PATH = "./data/indices/100k"
    DOCUMENTS_PATH = "./data/msmarco-docs.tsv.gz"
    MAX_DOCUMENTS = 100_000  # 3_213_835
    PARTITION_SIZE = 10_000
    DOCUMENTS_DATA_FILE_NAME = "documents.dat"
    DOCUMENTS_INDEX_FILE_NAME = "document.idx"
    POSTINGS_DATA_FILE_NAME = "postings.dat"
    POSTINGS_INDEX_FILE_NAME = "postings.idx"
    PARTITION_PREFIX = "part"
    TIER_PREFIX = "tier"
    NUM_FIELDS = 2
    FIELD_BOOSTS = [1.0, 0.5]
    NUM_TIERS = 4
    TIER_SCORE_THRESHOLDS = [20, 10, 5, 0]
    BM25_K = 1.5
    BM25_B_VALUES = [0.75, 0.75, 0.75, 0.75]
    SPELLING_FREQUENCY_THRESHOLD = 100


def main():

    tokenizer = Tokenizer()

    # indexer = Indexer(SEAConfig)
    # indexer.add_documents(
    #     load_documents(SEAConfig.DOCUMENTS_PATH, tokenizer, SEAConfig.MAX_DOCUMENTS)
    # )
    # indexer.merge_partitions()

    engine = Engine(SEAConfig)
    while True:
        query_text = input("Enter your search query: ")
        query = Query(query_text, tokenizer)
        start = time.time()
        results = engine.search(query, limit=10)
        end = time.time()
        print(f"Found {len(results)} results in {(end - start)*1000:.4f} milliseconds:")
        for score, doc in results:
            print(f"{score:.4f}: {doc}")


if __name__ == "__main__":
    main()
