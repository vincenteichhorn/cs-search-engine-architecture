import gzip
from tqdm import tqdm
from sea.document import Document
from sea.index import Index
from sea.query import Query
from sea.tokenizer import Tokenizer
from sea.util import load_documents

MAX_DOCUMENTS = 1000


def main():

    tokenizer = Tokenizer()
    documents = load_documents("./data/msmarco-docs.tsv.gz", tokenizer, max_documents=MAX_DOCUMENTS)
    print(len(documents), "documents loaded.")
    index = Index()
    index.add_documents(documents, verbose=True)
    print(index)

    while True:
        query_text = input("Enter your search query: ")
        query = Query(query_text, tokenizer)
        print(query)
        results = index.search(query)
        if results is []:
            print("No documents found matching the query.")
            continue
        else:
            print(f"Found {len(results)} documents matching the query.")
        for i, doc in enumerate(results):
            if i >= 10:
                break
            print(doc.title, doc.url)


if __name__ == "__main__":
    main()
