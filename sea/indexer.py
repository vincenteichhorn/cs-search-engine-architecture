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
        if os.path.exists(os.path.join(self.save_dir, self.posting_lists_file_name)):
            os.remove(os.path.join(self.save_dir, self.posting_lists_file_name))
        if os.path.exists(os.path.join(self.save_dir, self.posting_lists_index_file_name)):
            os.remove(os.path.join(self.save_dir, self.posting_lists_index_file_name))

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
            self.index = defaultdict(lambda: PostingList(key=lambda pst: pst.doc_id))
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

        def get_next_term_entry(posting_index_file):
            token_length_bytes = posting_index_file.read(4)
            token_length = struct.unpack(">I", token_length_bytes)[0]
            token_bytes = posting_index_file.read(token_length)
            token = token_bytes.decode("utf-8")
            offset = struct.unpack(">I", posting_index_file.read(4))[0]
            length = struct.unpack(">I", posting_index_file.read(4))[0]
            return token, token_bytes, offset, length

        def write_next_term_entry(posting_index_file, token_bytes, offset, length):
            token_length = len(token_bytes)
            posting_index_file.write(struct.pack(">I", token_length))
            posting_index_file.write(token_bytes)
            posting_index_file.write(struct.pack(">I", offset))
            posting_index_file.write(struct.pack(">I", length))

        if len(self.index) > 0:
            self.flush()

        partition_dirs = [
            os.path.join(self.save_dir, d)
            for d in os.listdir(self.save_dir)
            if os.path.isdir(os.path.join(self.save_dir, d)) and d.startswith("part")
        ]
        partition_dirs = sorted(partition_dirs, key=lambda x: int(x.split("part")[-1]))

        posting_lists_files = [
            open(os.path.join(part_dir, self.posting_lists_file_name), "rb")
            for part_dir in partition_dirs
        ]
        posting_indices = [
            open(os.path.join(part_dir, self.posting_lists_index_file_name), "rb")
            for part_dir in partition_dirs
        ]

        with (
            open(
                os.path.join(self.save_dir, self.posting_lists_file_name), "wb"
            ) as merged_posting_file,
            open(
                os.path.join(self.save_dir, self.posting_lists_index_file_name), "wb"
            ) as merged_posting_index_file,
        ):

            merged_offset = merged_length = 0

            candidate_partitions = []
            for pid, posting_list_index in enumerate(posting_indices):
                token, token_bytes, offset, length = get_next_term_entry(posting_list_index)
                candidate_partitions.append((pid, token, token_bytes, offset, length))

            sorted_candidates = sorted(candidate_partitions, key=lambda x: (x[1], x[0]))
            current_token = (
                sorted_candidates[0][1],  # token
                sorted_candidates[0][2],  # token_bytes
            )
            while True:

                pid, token, token_bytes, offset, length = sorted_candidates.pop(0)

                if current_token[0] != token:
                    write_next_term_entry(
                        merged_posting_index_file,
                        current_token[-1],
                        merged_offset,
                        merged_length,
                    )
                    merged_offset += merged_length
                    merged_length = 0

                current_token = (token, token_bytes)

                posting_lists_files[pid].seek(offset)
                posting_list_bytes = posting_lists_files[pid].read(length)
                merged_posting_file.write(posting_list_bytes)
                merged_length += length

                if posting_indices[pid].tell() == os.fstat(posting_indices[pid].fileno()).st_size:
                    if len(sorted_candidates):
                        continue
                    else:
                        write_next_term_entry(
                            merged_posting_index_file,
                            current_token[-1],
                            merged_offset,
                            merged_length,
                        )
                        break

                token, token_bytes, offset, length = get_next_term_entry(posting_indices[pid])

                new_candidate = (pid, token, token_bytes, offset, length)
                left, right = 0, len(sorted_candidates) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if (token, pid) < (sorted_candidates[mid][1], sorted_candidates[mid][0]):
                        right = mid - 1
                    else:
                        left = mid + 1
                sorted_candidates.insert(left, new_candidate)
