import gzip
import struct
from typing import List, Tuple

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


def encode_gamma(n: int):

    binary = bin(n)[3:]
    bytestr = "1" * len(binary) + "0" + binary
    return bytestr


def decode_gamma(bytestr: str):

    i = 0
    while i < len(bytestr) and bytestr[i] == "1":
        i += 1
    length = i
    binary = "1" + bytestr[i + 1 : i + 1 + length]
    return int(binary, 2), i + 1 + length


def pack_gammas(numbers: List[int]) -> bytearray:

    bits = ""
    for number in numbers:
        bits += encode_gamma(number)

    # Pad bits to make its length a multiple of 8
    while len(bits) % 8 != 0:
        bits += "0"
    bits += "0" * 8  # Extra byte to indicate end

    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        byte_array.append(int(byte, 2))

    return byte_array


def unpack_gammas(data: bytearray) -> Tuple[List[int], int]:

    bits = ""
    for byte in data:
        bits += f"{byte:08b}"

    numbers = []
    offset = 0
    while offset < len(bits) and "1" in bits[offset:]:
        n, length = decode_gamma(bits[offset:])
        numbers.append(n)
        offset += length
    return numbers, offset
