from collections import defaultdict
import os
import struct
from sea.document import Document
from sea.posting_list import PostingList


class Indexer:

    def __init__(self, save_dir: str, partition_size: int = 1000):
        self.partition_size = partition_size
        self.save_dir = save_dir
        self.index = defaultdict(lambda: PostingList(key=lambda pst: pst.doc_id))
        self.documents = []
        self.no_documents_in_partition = 0
        self.partition_id = 0

        os.makedirs(self.save_dir, exist_ok=True)

        self.documents_file_name = "documents.bin"
        self.document_index_file_name = "document_index.bin"
        self.posting_lists_file_name = "posting_lists.bin"
        self.posting_lists_index_file_name = "posting_lists_index.bin"

        if os.path.exists(os.path.join(self.save_dir, self.documents_file_name)):
            os.remove(os.path.join(self.save_dir, self.documents_file_name))
        if os.path.exists(os.path.join(self.save_dir, self.document_index_file_name)):
            os.remove(os.path.join(self.save_dir, self.document_index_file_name))

        for item in os.listdir(self.save_dir):
            item_path = os.path.join(self.save_dir, item)
            if os.path.isdir(item_path) and item.startswith("part"):
                for file in os.listdir(item_path):
                    os.remove(os.path.join(item_path, file))
                os.rmdir(item_path)

    def add_document(self, document: Document):

        self.documents.append(document)
        for token in document.tokens:
            self.index[token].add(document.get_posting(token))
        self.no_documents_in_partition += 1

        if self.no_documents_in_partition >= self.partition_size:
            self.flush()
            self.no_documents_in_partition = 0
            self.index = {}
            self.documents = []

    def flush(self):

        part_dir = os.path.join(self.save_dir, f"part{self.partition_id}")
        os.makedirs(part_dir, exist_ok=True)
        self.partition_id += 1

        with (
            open(os.path.join(self.save_dir, self.documents_file_name), "ab") as doc_file,
            open(
                os.path.join(self.save_dir, self.document_index_file_name), "ab"
            ) as doc_index_file,
        ):

            offset = doc_file.tell()
            for doc in self.documents:
                doc_bytes = doc.serialize()
                doc_file.write(doc_bytes)
                doc_index_file.write(struct.pack(">I", offset))
                doc_index_file.write(struct.pack(">I", len(doc_bytes)))
                offset += len(doc_bytes)

        sorted_tokens = sorted(self.index.keys())

        with (
            open(os.path.join(part_dir, self.posting_lists_file_name), "wb") as posting_file,
            open(
                os.path.join(part_dir, self.posting_lists_index_file_name), "wb"
            ) as posting_index_file,
        ):

            offset = 0
            for token in sorted_tokens:
                token_bytes = token.encode("utf-8")
                length = 0
                for posting in self.index[token]:
                    posting_bytes = posting.serialize()
                    posting_file.write(posting_bytes)
                    length += len(posting_bytes)
                posting_index_file.write(struct.pack(">I", len(token_bytes)))
                posting_index_file.write(token_bytes)
                posting_index_file.write(struct.pack(">I", offset))
                posting_index_file.write(struct.pack(">I", length))
                offset += length

    def merge_partitons(self):
        pass
