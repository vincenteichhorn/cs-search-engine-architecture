import os
from sea.indexer import Indexer
from sea.corpus import Corpus


INDEX_PATH = "./data/indices_new/10k"
DATASET = "./data/msmarco-docs.tsv"
MAX_DOCUMENTS = 10_000


def main():
    indexer = Indexer(INDEX_PATH, MAX_DOCUMENTS)
    corpus = Corpus(os.path.join(INDEX_PATH, "corpus"), DATASET)
    indexer.build(corpus)


if __name__ == "__main__":
    main()
