import gzip
from tqdm import tqdm
from sea.document import Document
from sea.index import Index
from sea.tokenizer import Tokenizer

MAX_DOCUMENTS = 500

if __name__ == "__main__":

    def load_documents(path):
        with gzip.open(path, 'rt', encoding='utf-8') as file: # 15.000 documente in memory laden
            incomplete_lines = 0
            documents = []

            for line in tqdm(file, desc="Loading documents", total=MAX_DOCUMENTS):
                if len(documents) >= MAX_DOCUMENTS:
                    break
                columns = line.strip().split("\t")
                if len(columns) < 4:
                    incomplete_lines += 1
                    continue  # weird lines skippen
                doc = Document(columns[2], columns[1], columns[3], tokenizer)
                documents.append(doc)

        print(f"Loaded {len(documents)} documents.")
        print("Weird lines with less than 4 columns (not read):", incomplete_lines)
        return documents

    tokenizer = Tokenizer()
    documents = load_documents("./data/msmarco-docs.tsv.gz")
    print(documents[0].tokens)
    index = Index()
    for doc in tqdm(documents, desc="Indexing documents"):
        index.add_document(doc)
    print(index) # ruft die __repr__() auf  
    

    while True:
        query = input("Enter your search query: ")
        print(f"Searching for: {query}")
        results = []
        query_tokens = tokenizer.tokenize_query(query)
        results = index.search(query_tokens)
        print(f"Found {len(results)} documents matching the query.")
        for i, doc in enumerate(results):
            if i >= 10:
                break
            print(doc.title, doc.url)

        