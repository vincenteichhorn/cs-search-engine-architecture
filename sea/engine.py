from collections import defaultdict
import os
import struct
from typing import List, Tuple

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

        with open(
            os.path.join(self.index_path, "part0/posting_lists_index.bin"), "rb"
        ) as f:

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
        with open(os.path.join(self.index_path, "part0/posting_lists.bin"), "rb") as f:
            f.seek(offset)
            posting_list_bytes = f.read(length)
        postings = []
        remainder_bytes = posting_list_bytes
        while int.from_bytes(remainder_bytes, "big") != 0:
            posting, remainder_bytes = Posting.deserialize(remainder_bytes)
            postings.append(posting)
        return PostingList.from_list(postings, key=lambda pst: pst.doc_id)

    def search(self, query: Query) -> List[Document]:
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
            while i < len(first_posting.positions) and j < len(
                second_posting.positions
            ):
                if first_posting.positions[i] + k == second_posting.positions[j]:
                    return True
                elif first_posting.positions[i] + k < second_posting.positions[j]:
                    i += 1
                else:
                    j += 1
            return False

        def evaluate_node(node: Node) -> Tuple[PostingList, bool]:
            if node.left is None and node.right is None:
                if isinstance(node.value, list):
                    result = self._get_postings(node.value[0]).clone()
                    previous_token = node.value[0]
                    for token in node.value[1:]:
                        other_posting_list = self._get_postings(token)
                        result.intersection(
                            other_posting_list, lambda a, b: contains_phrase(a, b, k=1)
                        ).clone()
                        previous_token = token
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
            raise NotImplementedError("NOT operator at the root is not supported yet.")
            # results = self._all_docs.clone().difference(results)

        return results
