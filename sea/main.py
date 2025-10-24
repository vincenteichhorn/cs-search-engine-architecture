from sea.document import Document
from sea.tokenizer import Tokenizer

if __name__ == "__main__":

    doc1 = Document("title", "www.", "Hello, we are hungry.")
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize_document(doc1)
    print(tokens)
    
    # def load_documents(path):
    # with gzip.open(path, 'rt', encoding='utf-8') as file: # 15.000 documente in memory laden
    #     count = 0
    #     weird = 0
    #     for line in file:
    #         if count >= max_documents:
    #             break
    #         columns = line.strip().split("\t")
    #         if len(columns) < 4:
    #             weird += 1
    #             continue  # weird lines skippen
    #         doc = {
    #             "docid": columns[0],
    #             "url": columns[1],
    #             "title": columns[2],
    #             "body": columns[3]}
    #         documents.append(doc)
    #         count += 1
    # print(f"Loaded {len(documents)} documents.")
    # print("Weird lines with less than 4 columns (not read):", weird)

    # while True:
    #     query = input("Enter your search query: ")
    #     print(f"Searching for: {query}")

        