from tqdm import tqdm
from sea.tokenizer import Tokenizer
from sea.engine import Engine
from sea.indexer import Indexer
from sea.query import Query
from sea.util.load import load_documents

MAX_DOCUMENTS = 100


def main():

    tokenizer = Tokenizer()

    documents = load_documents(
        "./data/msmarco-docs.tsv.gz", tokenizer, max_documents=MAX_DOCUMENTS
    )
    print(len(documents), "documents loaded.")

    indexer = Indexer("./data/index", partition_size=1500)
    for document in tqdm(documents, desc="Indexing documents"):
        indexer.add_document(document)
    indexer.flush()
    del indexer

    engine = Engine("./data/index")
    while True:
        query_text = input("Enter your search query: ")
        query = Query(query_text, tokenizer)
        print(query)
        results = engine.search(query)
        print(f"Found {len(results)} results:")


if __name__ == "__main__":
    main()
