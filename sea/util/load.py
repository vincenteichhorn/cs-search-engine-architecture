import gzip
import struct
from typing import List, Tuple, Union

from tqdm import tqdm
from sea.document import Document
from sea.tokenizer import Tokenizer


def load_documents(
    file_path: str, tokenizer: Tokenizer, max_documents: int = 500
) -> List[Document]:
    """
    Load documents from a gzipped TSV file. Each line in the file should contain
    at least four tab-separated columns: docid, url, title, and body.

    Args:
        file_path (str): Path to the gzipped TSV file.
        tokenizer (Tokenizer): An instance of the Tokenizer class to tokenize document text.
        max_documents (int, optional): Maximum number of documents to load (default is 500).

    Returns:
        List[Document]: A list of Document objects loaded from the file.
    """
    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        incomplete_lines = 0
        documents = []

        for line in tqdm(file, desc="Loading documents", total=max_documents):
            if len(documents) >= max_documents:
                break
            columns = line.strip().split("\t")
            if len(columns) < 4:
                incomplete_lines += 1
                continue
            doc = Document(columns[2], columns[1], columns[3], tokenizer)
            documents.append(doc)

    return documents
