from sea.posting import Posting
from sea.tokenizer import Tokenizer
from collections import defaultdict
from typing import List
import struct
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t

cdef int NEXT_ID = 1

cdef class Document:
    cdef public int id
    cdef public object title
    cdef public object url
    cdef public object body
    cdef public object tokenizer
    cdef public list tokens
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
        self.token_positions = None

    def __init__(self, title, url, body, tokenizer: Tokenizer = None):
        if tokenizer is not None:
            self._tokenize()
            self._compute_positions()

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
    
    def __repr__(self):
        return f"Document(id={self.id}, title={self.title}, url={self.url})"

    cdef void _tokenize(self):
        """
        Calls the tokenizer to tokenize the document's content
        """
        if self.tokens is None:
            self.tokens = self.tokenizer.tokenize_document(self)


    cdef void _compute_positions(self):
        """
        Counts the frequency of each token in the document
        """
        cdef object positions = {}
        cdef Py_ssize_t i
        cdef object token

        if self.token_positions is None and self.tokens is not None:
            for i, token in enumerate(self.tokens):
                if token not in positions:
                    positions[token] = []
                positions[token].append(i)

            self.token_positions = positions
        else:
            raise ValueError("Document must be tokenized before counting tokens.")


    cpdef bytes serialize(self):
        """
        Serialize document into a contiguous bytearray using preallocated buffer.
        """
        cdef bytes title_bytes = self.title.encode('utf-8')
        cdef bytes url_bytes = self.url.encode('utf-8')
        cdef bytes body_bytes = self.body.encode('utf-8')

        cdef Py_ssize_t total_size = 4 + 4 + len(title_bytes) + 4 + len(url_bytes) + 4 + len(body_bytes)
        cdef bytearray data = bytearray(total_size)
        cdef Py_ssize_t offset = 0

        struct.pack_into('>I', data, offset, self.id)
        offset += 4

        struct.pack_into('>I', data, offset, len(title_bytes))
        offset += 4
        data[offset:offset+len(title_bytes)] = title_bytes
        offset += len(title_bytes)

        struct.pack_into('>I', data, offset, len(url_bytes))
        offset += 4
        data[offset:offset+len(url_bytes)] = url_bytes
        offset += len(url_bytes)

        struct.pack_into('>I', data, offset, len(body_bytes))
        offset += 4
        data[offset:offset+len(body_bytes)] = body_bytes

        return bytes(data)

    @classmethod
    def deserialize(cls, bytearray data):
        """
        Deserialize a bytearray into a Document using struct.unpack_from (avoids slicing).
        """
        cdef Py_ssize_t offset = 0
        cdef int id, length
        cdef bytearray title_bytes, url_bytes, body_bytes

        id = struct.unpack_from('>I', data, offset)[0]
        offset += 4

        length = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        title_bytes = data[offset:offset+length]
        offset += length
        title = title_bytes.decode('utf-8')

        length = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        url_bytes = data[offset:offset+length]
        offset += length
        url = url_bytes.decode('utf-8')

        length = struct.unpack_from('>I', data, offset)[0]
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