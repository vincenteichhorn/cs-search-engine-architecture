import os
from sea.indexer import Indexer
from sea.corpus import Corpus


def main():
    INDEX_PATH = "./data/indices_new/10k"
    indexer = Indexer(INDEX_PATH)
    corpus = Corpus(os.path.join(INDEX_PATH, "corpus"), "./data/msmarco-docs.tsv")
    indexer.build(corpus)


if __name__ == "__main__":
    main()
