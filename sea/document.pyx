from sea.posting import Posting
from sea.tokenizer import Tokenizer
from collections import defaultdict
from typing import List
import struct
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t
from libc.stdint cimport uint8_t
from cpython.unicode cimport PyUnicode_DecodeUTF8

cdef int NEXT_ID = 1

cpdef document_deserialize(const uint8_t[:] data, cls=None):
    
    if cls is None:
        cls = Document

    cdef const uint8_t* data_ptr = &data[0]
    cdef Py_ssize_t offset = 0
    cdef int id, length
    cdef str title, url, body

    id = (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3]
    offset += 4

    length = (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3]
    offset += 4
    title = PyUnicode_DecodeUTF8(<char*>(data_ptr + offset), length, NULL)
    offset += length

    length = (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3]
    offset += 4
    url = PyUnicode_DecodeUTF8(<char*>(data_ptr + offset), length, NULL)
    offset += length

    length = (data[offset] << 24) | (data[offset + 1] << 16) | (data[offset + 2] << 8) | data[offset + 3]
    offset += 4
    body = PyUnicode_DecodeUTF8(<char*>(data_ptr + offset), length, NULL)

    doc = cls(title, url, body, None)
    doc.id = id
    global NEXT_ID
    NEXT_ID -= 1
    return doc

cdef class Document:
    cdef public int id
    cdef public str title
    cdef public str url
    cdef public str body
    cdef public object tokenizer
    cdef public list tokens
    cdef public object token_positions
    cdef public int num_title_tokens

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
        self.num_title_tokens = -1

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
            self.tokens, self.num_title_tokens = self.tokenizer.tokenize_document(self)


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
    def deserialize(cls, const uint8_t[:] data):
        """
        Deserialize a bytearray into a Document using struct.unpack_from (avoids slicing).
        """
        return document_deserialize(data)

    cpdef object get_posting(self, str token):
        """
        Returns a Posting object for the given token in this document.
        """
        positions = self.get_token_positions(token)
        tf_body = tf_title = 0
        for pos in positions:
            if pos < self.num_title_tokens:
                tf_title += 1
            else:
                tf_body += 1
        # id, positions, term frequencies per field, length of each field
        return Posting(self.id, positions, [tf_title, tf_body], [self.num_title_tokens, len(self.tokens)-self.num_title_tokens])