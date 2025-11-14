import time
from tqdm import tqdm

from sea.tokenizer import Tokenizer
from sea.util.load import load_documents

MAX_DOCUMENTS = 1000

if __name__ == "__main__":

    tokenizer = Tokenizer()

    start = time.time()
    times = []

    for document in tqdm(
        load_documents("./data/msmarco-docs.tsv.gz", tokenizer, max_documents=MAX_DOCUMENTS),
        desc="Indexing documents",
        total=MAX_DOCUMENTS,
    ):
        end = time.time()
        times.append(end - start)
        start = time.time()

    print(f"Avg: {(sum(times)/len(times))*1000:.4f} ms per document")
