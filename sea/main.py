import os
import time
from sea.indexer import Indexer
from sea.corpus import Corpus, py_document_processor
from sea.engine import Engine
import shutil

INDEX_PATH = "./data/indices_new/all"
DATASET = "./data/msmarco-docs.tsv"  # "./data/testing_merge.tsv"
MAX_DOCUMENTS = 3_213_835  # 3_213_835
PARTITION_SIZE = 20_000


def bold_string(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def index():

    inp = input(
        f"Do you really want to (re)build the index at {INDEX_PATH}? This will delete any existing index. (y/n): "
    )
    if inp.lower() != "y":
        return
    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH, ignore_errors=True)

    indexer = Indexer(INDEX_PATH, DATASET, MAX_DOCUMENTS, PARTITION_SIZE)
    indexer.build()


def serve():

    engine = Engine(INDEX_PATH)

    while True:
        query = input("Search: ")
        start = time.time()
        results = engine.search(query, top_k=10)
        end = time.time()
        print(f"Search took {(end - start) * 1000:.4f} milliseconds.")
        for doc in results:
            print("-" * os.get_terminal_size().columns)
            print(
                f"{bold_string('Title')}: {doc['title']}",
                f"{bold_string('URL')}: {doc['url']}",
                f"{bold_string('Score')}: {doc['score']:.4f}",
                f"{bold_string('Snippet')}: {doc['snippet']}\n",
                sep="\n",
            )


def main():
    index()
    serve()


if __name__ == "__main__":
    main()
