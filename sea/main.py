import os
from sea.indexer import Indexer
from sea.corpus import Corpus, py_document_processor
import shutil

INDEX_PATH = "./data/indices_new/10k"
DATASET = "./data/msmarco-docs.tsv"
MAX_DOCUMENTS = 1_000


def main():

    if os.path.exists(INDEX_PATH):
        shutil.rmtree(INDEX_PATH, ignore_errors=True)

    corpus = Corpus(os.path.join(INDEX_PATH, "corpus"), DATASET)
    indexer = Indexer(INDEX_PATH, corpus, MAX_DOCUMENTS)
    indexer.build()

    del indexer
    del corpus

    corpus = Corpus(os.path.join(INDEX_PATH, "corpus"), DATASET)
    print(corpus.py_get(1, py_document_processor))


if __name__ == "__main__":
    main()
