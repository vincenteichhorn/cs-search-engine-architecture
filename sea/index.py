from collections import defaultdict
from typing import List

from tqdm import tqdm
from sea.posting_list import PostingList
from sea.document import Document


class Index:
    def __init__(self):

        def new_posting_list():
            return PostingList(key=lambda doc: doc.id)

        self._index = defaultdict(new_posting_list)
        self._doc_counts = defaultdict(int)

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

    def search(self, query_tokens: List[str]) -> List[Document]:
        """
        Search the index for documents matching the given query tokens.
        Args:
            query_tokens (List[str]): A list of tokens representing the search query.

        Returns:
            List[Document]: A list of Document objects that match the query.
        """
        if not query_tokens:
            return []

        results = None
        for token in query_tokens:
            if token in self._index:
                if results is None:
                    results = self._index[token].clone()
                else:
                    results = results.intersection(self._index[token])

        return results

    def __repr__(self):
        return f"Index(num_terms={len(self._index)})"
