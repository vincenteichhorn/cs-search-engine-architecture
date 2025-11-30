from sea.posting import Posting
from sea.tokenizer import Tokenizer
import struct
from libc.stdint cimport uint8_t
from cpython.unicode cimport PyUnicode_DecodeUTF8
from array import array

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
    cdef public list char_positions
    cdef public object token_char_positions
    cdef public int num_title_chars
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
        self.char_positions = None
        self.num_title_chars = -1
        self.num_title_tokens = -1

    def __init__(self, title, url, body, tokenizer: Tokenizer = None):
        self.title = title
        self.url = url
        self.body = body
        self.tokenizer = tokenizer
    
    def __repr__(self):
        return f"Document(id={self.id}, title={self.title}, url={self.url})"

    cpdef void ensure_tokenized(self):
        """
        Ensures that the document is tokenized
        """
        if self.tokens is None:
            self._tokenize()
            self._compute_positions()

    cdef void _tokenize(self):
        """
        Calls the tokenizer to tokenize the document's content
        """
        if self.tokens is not None:
            return

        cdef list title_toks
        cdef list char_title_positions
        self.tokens, self.char_positions = self.tokenizer.tokenize(self.title, False)
        self.num_title_tokens = len(self.tokens)
        self.num_title_chars = len(self.title)
        cdef list body_toks
        cdef list char_body_positions
        body_toks, char_body_positions = self.tokenizer.tokenize(self.body, False)
        self.tokens.extend(body_toks)
        self.char_positions.extend(char_body_positions)

    cpdef list get_tokens_unique(self):
        """
        Returns the tokens of the document, tokenizing if necessary
        """
        self.ensure_tokenized()
        if self.token_char_positions is not None:
            return list(self.token_char_positions.keys())
        raise ValueError("Document must be tokenized before getting unique tokens.")

    cdef void _compute_positions(self):
        """
        Counts the frequency of each token in the document
        """
        cdef Py_ssize_t i, n = len(self.tokens)
        cdef object token
        cdef object char_position
        cdef object token_char_positions = {}

        if self.token_char_positions is None and self.tokens is not None:
            for i in range(n):
                token = self.tokens[i]
                char_position = self.char_positions[i]
                lst = token_char_positions.get(token)
                if lst is None:
                    lst = []
                    token_char_positions[token] = lst
                lst.append(char_position)

            self.token_char_positions = token_char_positions
        else:
            raise ValueError("Document must be tokenized before counting tokens.")


    cpdef bytes serialize(self):
        """
        Serialize document into a contiguous bytearray using preallocated buffer.
        """
        self.ensure_tokenized()
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
        self.ensure_tokenized()
        cdef list char_positions = self.token_char_positions.get(token, [])
        cdef int tf_body = 0
        cdef int tf_title = 0
        for pos in char_positions:
            if pos < self.num_title_chars:
                tf_title += 1
            else:
                tf_body += 1
        # id, positions, term frequencies per field, length of each field
        return Posting(self.id, char_positions, [tf_title, tf_body], [self.num_title_tokens, len(self.tokens)-self.num_title_tokens])