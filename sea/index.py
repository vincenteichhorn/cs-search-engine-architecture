from collections import defaultdict
from sea.posting_list import PostingList


class Index:
    def __init__(self):

        def new_posting_list():
            return PostingList(key=lambda doc: doc.id)
        
        self._index = defaultdict(new_posting_list)
        self._doc_counts = defaultdict(int)

    def add_document(self, document):
        if document.tokens is None:
            raise ValueError("Document must be tokenized before adding to index.")
        for token in document.tokens:
            self._index[token].add(document)
            self._doc_counts[token] += 1

    def search(self, query_tokens):
        if not query_tokens:
            return []
        
        results = None

        for token in query_tokens:
            if token in self._index:
                if results is None:
                    results = self._index[token].clone() #damit das Objekt im Index nicht geändert wird
                else:
                    results = results.intersection(self._index[token])

        return results

                

    def __repr__(self):
        return f"Index(num_terms={len(self._index)})"