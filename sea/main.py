import os
import time
from sea.indexer import Indexer
from sea.corpus import Corpus, py_document_processor
from sea.engine import Engine
import shutil

INDEX_PATH = "./data/indices_new/100k"
DATASET = "./data/msmarco-docs.tsv"  # "./data/testing_merge.tsv"
MAX_DOCUMENTS = 100_000
PARTITION_SIZE = 20_000


def bold_string(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def index():

    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH, ignore_errors=True)

    indexer = Indexer(INDEX_PATH, DATASET, MAX_DOCUMENTS, PARTITION_SIZE)
    indexer.build()


def serve():

    engine = Engine(INDEX_PATH)

    while True:
        query = input("Enter your query: ")
        start = time.time()
        results = engine.search(query, top_k=10)
        end = time.time()
        print(f"Search took {(end - start) * 1000:.4f} milliseconds.")
        print("Top results:")
        for doc in results:
            print("-" * os.get_terminal_size().columns)
            print(
                f"{bold_string('Title')}: {doc['title']}",
                f"{bold_string('URL')}: {doc['url']}",
                f"{bold_string('Score')}: {doc['score']:.4f}",
                sep="\n",
            )
            print(f"{bold_string('Snippet')}: {doc['snippet']}\n")


def main():
    # index()
    serve()


if __name__ == "__main__":
    main()
