import gzip
import time
from sea.tokenizer import Tokenizer
from sea.document import Document

MAX_DOCUMENTS = 100

if __name__ == "__main__":

    tokenizer = Tokenizer()

    file_path = "./data/msmarco-docs.tsv.gz"
    open_file_fun = gzip.open if file_path.endswith(".gz") else open

    times = []

    with open_file_fun(file_path, "rt", encoding="utf-8") as file:
        incomplete_lines = 0
        count = 0
        for line in file:
            if count >= MAX_DOCUMENTS:
                break
            columns = line.strip().split("\t")
            if len(columns) < 4:
                incomplete_lines += 1
                continue
            start = time.time()
            doc = Document(columns[2], columns[1], columns[3], tokenizer)
            end = time.time()
            times.append(end - start)
            count += 1
            break
    print(f"Avg: {(sum(times)/len(times))*1000:.4f} ms per document")
