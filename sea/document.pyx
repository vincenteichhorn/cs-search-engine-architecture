from sea.posting import Posting
from sea.tokenizer import Tokenizer
from collections import defaultdict
from typing import List
import struct

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

    def __cinit__(self, title, url, body, tokenizer: Tokenizer = None):
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

    def __init__(self, title, url, body, tokenizer: Tokenizer = None):
        if tokenizer is not None:
            self._tokenize()
            self._count_tokens()

    def get_token_positions(self, token: str) -> List[int]:
        """
        Returns the list of positions for a given token in the document.

        Arguments:
            token (str): The token to retrieve positions for.

        Returns:
            List[int]: A list of positions where the token appears in the document.
        """
        if self.token_positions is not None and token in self.token_positions:
            return self.token_positions[token]
        else:
            return []
    
    def get_term_frequency(self, token: str) -> int:
        """
        Returns the frequency of a given token in the document.

        Arguments:
            token (str): The token to retrieve frequency for.

        Returns:
            int: The frequency of the token in the document.
        """
        if self.token_counts is not None and token in self.token_counts:
            return self.token_counts[token]
        else:
            return 0

    def __repr__(self):
        return f"Document(id={self.id}, title={self.title}, url={self.url})"

    cdef void _tokenize(self):
        """
        Calls the tokenizer to tokenize the document's content
        """
        if self.tokens is None:
            # Tokenizer returns a list of tokens
            self.tokens = self.tokenizer.tokenize_document(self)

    cdef void _count_tokens(self):
        """
        Counts the frequency of each token in the document
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

    cpdef bytearray serialize(self):
        """
        Serializes the Document into a bytearray.
        """
        cdef bytearray data = bytearray()
        cdef bytes title_bytes = self.title.encode('utf-8')
        cdef bytes url_bytes = self.url.encode('utf-8')
        cdef bytes body_bytes = self.body.encode('utf-8')

        data.extend(struct.pack('>I', self.id))
        data.extend(struct.pack('>I', len(title_bytes)))
        data.extend(title_bytes)
        data.extend(struct.pack('>I', len(url_bytes)))
        data.extend(url_bytes)
        data.extend(struct.pack('>I', len(body_bytes)))
        data.extend(body_bytes)

        return data

    @classmethod
    def deserialize(cls, bytearray data):
        """
        Deserializes a bytearray into a Document.
        """
        cdef Py_ssize_t offset = 0
        cdef int length
        cdef bytearray title_bytes
        cdef bytearray url_bytes
        cdef bytearray body_bytes
        cdef str title
        cdef str url
        cdef str body
        cdef int id

        id = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4

        length = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        title_bytes = data[offset:offset+length]
        offset += length

        title = title_bytes.decode('utf-8')
        length = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        url_bytes = data[offset:offset+length]
        offset += length

        url = url_bytes.decode('utf-8')
        length = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        body_bytes = data[offset:offset+length]
        body = body_bytes.decode('utf-8')

        doc = cls(title, url, body, None)
        doc.id = id
        global NEXT_ID
        NEXT_ID -= 1 
        return doc

    cpdef object get_posting(self, str token):
        """
        Returns a Posting object for the given token in this document.
        """
        return Posting(self.id, self.get_token_positions(token))