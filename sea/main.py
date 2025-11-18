import time
from tqdm import tqdm
from sea.tokenizer import Tokenizer
from sea.engine import Engine
from sea.indexer import Indexer
from sea.query import Query
from sea.util.load import load_documents

MAX_DOCUMENTS = 100_000  # 3_213_835


def main():

    tokenizer = Tokenizer()

    # indexer = Indexer("./data/indices/small", partition_size=7500)
    # start = time.time()
    # for document in tqdm(
    #     load_documents("./data/msmarco-docs.tsv.gz", tokenizer, max_documents=MAX_DOCUMENTS),
    #     desc="Indexing documents",
    #     total=MAX_DOCUMENTS,
    # ):
    #     indexer.add_document(document)
    # indexer.merge_partitions()
    # del indexer

    engine = Engine("./data/indices/small")
    while True:
        query_text = input("Enter your search query: ")
        start = time.time()
        query = Query(query_text, tokenizer)
        end = time.time()
        print(f"- Query processing took {(end - start)*1000:.4f} milliseconds")
        print(query)
        start = time.time()
        results = engine.search(query, limit=10)
        end = time.time()
        print(f"Found {len(results)} results in {(end - start)*1000:.4f} milliseconds:")
        for doc in results:
            print(doc)


if __name__ == "__main__":
    main()
