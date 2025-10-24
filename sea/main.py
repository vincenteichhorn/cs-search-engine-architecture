import gzip
from sea.document import Document
from sea.tokenizer import Tokenizer

MAX_DOCUMENTS = 1000

if __name__ == "__main__":

    def load_documents(path):
        with gzip.open(path, 'rt', encoding='utf-8') as file: # 15.000 documente in memory laden
            incomplete_lines = 0
            documents = []

            for line in file:
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
    print(documents[0].token_counts)
    

    # while True:
    #     query = input("Enter your search query: ")
    #     print(f"Searching for: {query}")

        