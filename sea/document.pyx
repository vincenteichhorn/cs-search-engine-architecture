from collections import defaultdict
from typing import List

cdef int NEXT_ID = 1

cdef class Document:
    cdef public int id
    cdef public object title
    cdef public object url
    cdef public object body
    cdef public object tokenizer
    cdef public list tokens
    cdef public object token_counts
    cdef public object token_positions

    def __cinit__(self, title, url, body, tokenizer):
        self.title = title
        self.url = url
        self.body = body
        self.tokenizer = tokenizer
        global NEXT_ID
        self.id = NEXT_ID
        NEXT_ID += 1
        self.tokens = None
        self.token_counts = None
        self.token_positions = None

    def __init__(self, title, url, body, tokenizer):
        self._tokenize()
        self._count_tokens()

    def __repr__(self):
        return f"Document(id={self.id}, title={self.title}, url={self.url})"

    cdef void _tokenize(self):
        """
        ruft den Tokenizer auf, um das Dokument zu tokenisieren
        """
        if self.tokens is None:
            # Tokenizer returns a list of tokens
            self.tokens = self.tokenizer.tokenize_document(self)

    cdef void _count_tokens(self):
        """
        Zählt die Häufigkeit jedes Tokens im Dokument
        """
        cdef object counts = defaultdict(int)
        cdef object positions = defaultdict(list)
        cdef Py_ssize_t i
        cdef object token

        if self.token_counts is None and self.tokens is not None:
            for i, token in enumerate(self.tokens):
                counts[token] += 1
                positions[token].append(i)

            self.token_counts = counts
            self.token_positions = positions
        else:
            raise ValueError("Document must be tokenized before counting tokens.")

