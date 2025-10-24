from collections import defaultdict


class Index:
    def __init__(self):
        self._index = defaultdict(lambda: list)

    def add_document(self, document):
        if document.tokens is None:
            raise ValueError("Document must be tokenized before adding to index.")
        for token in document.tokens:
            self._index[token].append(document)