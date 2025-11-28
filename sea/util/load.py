import gzip
from typing import Generator

from sea.document import Document
from sea.tokenizer import Tokenizer
import subprocess


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return subprocess.Popen(["zcat", path], stdout=subprocess.PIPE, text=True).stdout
    else:
        return open(path, "r", encoding="utf-8")


def load_documents(
    file_path: str, tokenizer: Tokenizer, max_documents: int = 500
) -> Generator[Document, None, None]:
    """
    Load documents from a gzipped TSV file. Each line in the file should contain
    at least four tab-separated columns: docid, url, title, and body.

    Args:
        file_path (str): Path to the gzipped TSV file.
        tokenizer (Tokenizer): An instance of the Tokenizer class to tokenize document text.
        max_documents (int, optional): Maximum number of documents to load (default is 500).

    Returns:
        Generator[Document, None, None]: A generator that yields Document objects loaded from the file.
    """

    with open_maybe_gzip(file_path) as file:
        incomplete_lines = 0
        count = 0
        for line in file:
            if count >= max_documents:
                break
            columns = line.strip().split("\t")
            if len(columns) < 4:
                incomplete_lines += 1
                continue
            doc = Document(columns[2], columns[1], columns[3], tokenizer)
            count += 1
            yield doc
