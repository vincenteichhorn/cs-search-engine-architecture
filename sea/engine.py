from collections import defaultdict
import os
import struct
from typing import List, Tuple, Union

from tqdm import tqdm
from sea.posting_list import PostingList
from sea.document import Document
from sea.posting import Posting
from sea.query import Node, Query


class Engine:
    def __init__(self, index_path: str):

        self.index_path = index_path
        self.token_dictionary = {}
        self._load_token_dictionary()

    def _load_token_dictionary(self) -> None:
        """
        Load the token dictionary from the index files.
        """

        with open(os.path.join(self.index_path, "posting_lists_index.bin"), "rb") as f:

            offset = 0
            while True:
                token_length_bytes = f.read(4)
                if not token_length_bytes:
                    break
                token_length = struct.unpack(">I", token_length_bytes)[0]
                token_bytes = f.read(token_length)
                token = token_bytes.decode("utf-8")
                posting_list_offset_bytes = f.read(4)
                posting_list_offset = struct.unpack(">I", posting_list_offset_bytes)[0]
                posting_list_length_bytes = f.read(4)
                posting_list_length = struct.unpack(">I", posting_list_length_bytes)[0]
                self.token_dictionary[token] = (
                    posting_list_offset,
                    posting_list_length,
                )

    def _get_postings(self, token: str) -> PostingList:
        """
        Retrieve the posting list for a given token from the index files.
        Args:
            token (str): The token to retrieve the posting list for.
        """
        if token not in self.token_dictionary:
            return PostingList(key=lambda doc: doc.id)

        offset, length = self.token_dictionary[token]
        with open(os.path.join(self.index_path, "posting_lists.bin"), "rb") as f:
            f.seek(offset)
            posting_list_bytes = f.read(length)
        postings = []
        remainder_bytes = posting_list_bytes
        while int.from_bytes(remainder_bytes, "big") != 0:
            posting, remainder_bytes = Posting.deserialize(remainder_bytes)
            postings.append(posting)
        return PostingList.from_list(postings, key=lambda pst: pst.doc_id)

    def _get_document(self, doc_id: int) -> Document:

        with (
            open(os.path.join(self.index_path, "document_index.bin"), "rb") as doc_index_file,
            open(os.path.join(self.index_path, "documents.bin"), "rb") as doc_file,
        ):
            index_offset = (doc_id - 1) * 8
            doc_index_file.seek(index_offset)
            offset_bytes = doc_index_file.read(4)
            offset = struct.unpack(">I", offset_bytes)[0]
            length_bytes = doc_index_file.read(4)
            length = struct.unpack(">I", length_bytes)[0]
            doc_file.seek(offset)
            doc_bytes = doc_file.read(length)
            doc = Document.deserialize(bytearray(doc_bytes))
            return doc

    def _get_not_documents(self, exclude_ids: List[int], limit: int = 10):

        index_offset = 0
        docs = []

        with (
            open(os.path.join(self.index_path, "document_index.bin"), "rb") as doc_index_file,
            open(os.path.join(self.index_path, "documents.bin"), "rb") as doc_file,
        ):
            while len(docs) < limit:
                doc_index_file.seek(index_offset)
                offset_bytes = doc_index_file.read(4)
                offset = struct.unpack(">I", offset_bytes)[0]
                length_bytes = doc_index_file.read(4)
                length = struct.unpack(">I", length_bytes)[0]
                doc_file.seek(offset)
                doc_bytes = doc_file.read(length)
                doc = Document.deserialize(bytearray(doc_bytes))
                if doc.id not in exclude_ids:
                    docs.append(doc)
                index_offset += 8
        return docs

    def search(self, query: Query, limit: int = 10) -> List[Document]:
        """
        Search the index for documents matching the given query tokens.
        Args:
            query_tokens (List[str]): A list of tokens representing the search query.

        Returns:
            List[Document]: A list of Document objects that match the query.
        """
        if query.root is None:
            return []

        def contains_phrase(first_posting, second_posting, k=1):
            i = j = 0
            while i < len(first_posting.positions) and j < len(second_posting.positions):
                if first_posting.positions[i] + k == second_posting.positions[j]:
                    return True
                elif first_posting.positions[i] + k < second_posting.positions[j]:
                    i += 1
                else:
                    j += 1
            return False

        def evaluate_node(node: Union[Node, None]) -> Tuple[PostingList, bool]:
            if node is None:
                return PostingList(key=lambda pst: pst.doc_id), False

            if node.left is None and node.right is None:
                if isinstance(node.value, list):
                    result = self._get_postings(node.value[0]).clone()
                    for token in node.value[1:]:
                        other_posting_list = self._get_postings(token)
                        result.intersection(
                            other_posting_list, lambda a, b: contains_phrase(a, b, k=1)
                        ).clone()
                    return result, False

                else:
                    return (
                        self._get_postings(node.value).clone(),
                        False,
                    )

            if node.value == "not":
                right_postings, right_is_not = evaluate_node(node.right)
                return right_postings, not right_is_not

            left_postings, left_is_not = evaluate_node(node.left)
            right_postings, right_is_not = evaluate_node(node.right)

            if node.value == "and":

                if not left_is_not and not right_is_not:
                    return left_postings.intersection(right_postings), False

                elif left_is_not and not right_is_not:
                    return right_postings.difference(left_postings), False

                elif not left_is_not and right_is_not:
                    return left_postings.difference(right_postings), False

                else:
                    return left_postings.union(right_postings), True

            elif node.value == "or":

                if not left_is_not and not right_is_not:
                    return left_postings.union(right_postings), False

                elif left_is_not and not right_is_not:
                    return left_postings.difference(right_postings), True

                elif not left_is_not and right_is_not:
                    return right_postings.difference(left_postings), True

                else:
                    return left_postings.intersection(right_postings), True
            else:
                raise ValueError(f"Unknown operator: {node.value}")

        results, is_not = evaluate_node(query.root)

        if is_not:
            return self._get_not_documents([res.doc_id for res in results], limit)

        docs = []
        for res in results:
            if len(docs) == limit:
                break
            docs.append(self._get_document(res.doc_id))
        return docs
