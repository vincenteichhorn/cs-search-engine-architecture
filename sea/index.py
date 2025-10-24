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
            self._index[token].append(document)
            self._doc_counts[token] += 1