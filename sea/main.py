import gzip
from tqdm import tqdm
from sea.document import Document
from sea.engine import Engine
from sea.indexer import Indexer
from sea.query import Query
from sea.tokenizer import Tokenizer
from sea.util import load_documents

MAX_DOCUMENTS = 1000


def main():

    tokenizer = Tokenizer()
    # documents = load_documents("./data/msmarco-docs.tsv.gz", tokenizer, max_documents=MAX_DOCUMENTS)
    # print(len(documents), "documents loaded.")

    # indexer = Indexer("./data/index", partition_size=1000)
    # for document in tqdm(documents, desc="Indexing documents"):
    #     indexer.add_document(document)
    # indexer.flush()

    engine = Engine("./data/index")
    while True:
        query_text = input("Enter your search query: ")
        query = Query(query_text, tokenizer)
        print(query)
        results = engine.search(query)
        print(results)


if __name__ == "__main__":
    main()
