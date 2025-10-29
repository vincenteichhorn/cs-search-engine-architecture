import gzip
from tqdm import tqdm
from sea.document import Document
from sea.index import Index
from sea.tokenizer import Tokenizer
from sea.util import load_documents

MAX_DOCUMENTS = 500


def main():

    tokenizer = Tokenizer()
    documents = load_documents("./data/msmarco-docs.tsv.gz", tokenizer, max_documents=MAX_DOCUMENTS)
    index = Index()
    index.add_documents(documents, verbose=True)
    print(index)

    while True:
        query = input("Enter your search query: ")
        query_tokens = tokenizer.tokenize_query(query)
        results = index.search(query_tokens)
        print(f"Found {len(results)} documents matching the query.")
        for i, doc in enumerate(results):
            if i >= 10:
                break
            print(doc.title, doc.url)


if __name__ == "__main__":
    main()
