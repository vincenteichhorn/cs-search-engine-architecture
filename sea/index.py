from collections import defaultdict
from typing import List, Tuple

from tqdm import tqdm
from sea.posting_list import PostingList
from sea.document import Document
from sea.query import Node, Query


class Index:
    def __init__(self):

        def new_posting_list():
            return PostingList(key=lambda doc: doc.id)

        self._index = defaultdict(new_posting_list)
        self._doc_counts = defaultdict(int)
        self._all_docs = PostingList(key=lambda doc: doc.id)

    def add_document(self, document: Document) -> None:
        """
        Add a single document to the index.

        Args:
            document (Document): The Document object to be added to the index.
        """
        if document.tokens is None:
            raise ValueError("Document must be tokenized before adding to index.")
        for token in document.tokens:
            self._index[token].add(document)
            self._all_docs.add(document)
            self._doc_counts[token] += 1

    def add_documents(self, documents: List[Document], verbose: bool = False) -> None:
        """
        Add multiple documents to the index.

        Args:
            documents (List[Document]): A list of Document objects to be added to the index.
            verbose (bool): If True, print progress messages.
        """
        for document in tqdm(documents, disable=not verbose, desc="Indexing documents"):
            self.add_document(document)

    def search(self, query: Query ) -> List[Document]:
        """
        Search the index for documents matching the given query tokens.
        Args:
            query_tokens (List[str]): A list of tokens representing the search query.

        Returns:
            List[Document]: A list of Document objects that match the query.
        """
        if query.root is None:
            return []
        
        def evaluate_node(node: Node) -> Tuple[PostingList, bool]:
            if node.left is None and node.right is None:
                return self._index.get(node.value, PostingList(key=lambda doc: doc.id)).clone(), node.is_not
            
            left_postings, left_is_not = evaluate_node(node.left)
            right_postings, right_is_not = evaluate_node(node.right)

            if node.value == "and":

                if not left_is_not and not right_is_not:
                    return left_postings.union(right_postings), False
                
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
        
        results, is_not = evaluate_node(query.root)
        if is_not:
            results = self._all_docs.difference(results)
        
        return results

    def __repr__(self):
        return f"Index(num_terms={len(self._index)})"
