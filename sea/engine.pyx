import mmap
import os
import struct
import time

from sea.posting_list import PostingList, posting_list_from_list
from sea.document import Document, document_deserialize
from sea.posting import Posting
from sea.util.gamma import BitReader
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t
from libc.stdint cimport uint8_t, uint64_t
from cpython.unicode cimport PyUnicode_DecodeUTF8
from sea.posting import posting_deserialize

def doc_id_key(doc):
    return doc.id

def pst_id_key(pst):
    return pst.doc_id


cdef class Engine:
    cdef public str index_path
    cdef object posting_lists_file
    cdef object posting_list_mmap
    cdef const uint8_t[:] posting_list_view
    cdef const uint8_t* posting_list_ptr

    cdef object posting_index_file
    cdef object posting_list_index_mmap
    cdef const uint8_t[:] posting_list_index_view
    cdef const uint8_t* posting_list_index_ptr

    cdef object document_index_file
    cdef object document_index_mmap
    cdef const uint8_t[:] document_index_view
    cdef const uint8_t* document_index_ptr

    cdef object documents_file
    cdef object documents_mmap
    cdef const uint8_t[:] documents_view
    cdef const uint8_t* documents_ptr

    cdef dict token_dictionary

    def __init__(self, str index_path):
        self.index_path = index_path

        self.posting_lists_file = open(os.path.join(self.index_path, "posting_lists.bin"), "rb")
        self.posting_list_mmap = mmap.mmap(self.posting_lists_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.posting_list_view = self.posting_list_mmap
        self.posting_list_ptr = &self.posting_list_view[0]
        
        self.posting_index_file = open(os.path.join(self.index_path, "posting_lists_index.bin"), "rb")
        self.posting_list_index_mmap = mmap.mmap(self.posting_index_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.posting_list_index_view = self.posting_list_index_mmap
        self.posting_list_index_ptr = &self.posting_list_index_view[0]

        self.document_index_file = open(os.path.join(self.index_path, "document_index.bin"), "rb")
        self.document_index_mmap = mmap.mmap(self.document_index_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.document_index_view = self.document_index_mmap
        self.document_index_ptr = &self.document_index_view[0]

        self.documents_file = open(os.path.join(self.index_path, "documents.bin"), "rb")
        self.documents_mmap = mmap.mmap(self.documents_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.documents_view = self.documents_mmap
        self.documents_ptr = &self.documents_view[0]

        self.token_dictionary = {}
        cdef clock_t start = clock()
        self._load_token_dictionary()
        cdef clock_t end = clock()
        cdef double elapsed = (end - start) / CLOCKS_PER_SEC * 1000
        print(f"Loading token dictionary took {elapsed:.4f} milliseconds")

    cpdef void _load_token_dictionary(self):
        cdef unsigned int file_size = self.posting_list_index_view.shape[0]
        cdef unsigned int pos = 0
        cdef str token
        cdef unsigned long long posting_list_offset
        cdef unsigned int posting_list_length
        while pos < file_size:
            token_length = (self.posting_list_index_view[pos] << 24) | (self.posting_list_index_view[pos + 1] << 16) | (self.posting_list_index_view[pos + 2] << 8) | self.posting_list_index_view[pos + 3]
            pos += 4
            token = PyUnicode_DecodeUTF8(<char*>(self.posting_list_index_ptr + pos), token_length, NULL)
            pos += token_length
            posting_list_offset = (<uint64_t>self.posting_list_index_view[pos] << 56) | (<uint64_t>self.posting_list_index_view[pos + 1] << 48) | (<uint64_t>self.posting_list_index_view[pos + 2] << 40) | (<uint64_t>self.posting_list_index_view[pos + 3] << 32) | (<uint64_t>self.posting_list_index_view[pos + 4] << 24) | (<uint64_t>self.posting_list_index_view[pos + 5] << 16) | (<uint64_t>self.posting_list_index_view[pos + 6] << 8) | <uint64_t>self.posting_list_index_view[pos + 7]
            pos += 8
            posting_list_length = (self.posting_list_index_view[pos] << 24) | (self.posting_list_index_view[pos + 1] << 16) | (self.posting_list_index_view[pos + 2] << 8) | self.posting_list_index_view[pos + 3]
            pos += 4
            self.token_dictionary[token] = (posting_list_offset, posting_list_length)

    cpdef object _get_postings(self, str token, bint load_positions=False):
        if token not in self.token_dictionary:
            return PostingList(key=pst_id_key)

        cdef long long offset
        cdef int length
        offset, length = self.token_dictionary[token]
        cdef list postings = []
        cdef object posting
        cdef Py_ssize_t bytes_read
        cdef Py_ssize_t end_offset = offset + length
        cdef Py_ssize_t cur = offset
        while cur < end_offset:
            posting, bytes_read = posting_deserialize(self.posting_list_view[cur:end_offset], only_doc_id=not load_positions)
            cur += bytes_read
            postings.append(posting)

        cdef object posting_list = posting_list_from_list(postings, key=pst_id_key, sorted=True)

        return posting_list

    cpdef object _get_document(self, int doc_id):
        cdef long long index_offset = (doc_id - 1) * 12
        cdef long long offset = (<uint64_t>self.document_index_view[index_offset] << 56) | (<uint64_t>self.document_index_view[index_offset+1] << 48) | (<uint64_t>self.document_index_view[index_offset+2] << 40) | (<uint64_t>self.document_index_view[index_offset+3] << 32) | (<uint64_t>self.document_index_view[index_offset+4] << 24) | (<uint64_t>self.document_index_view[index_offset+5] << 16) | (<uint64_t>self.document_index_view[index_offset+6] << 8) | <uint64_t>self.document_index_view[index_offset+7]
        cdef int length = (self.document_index_view[index_offset+8] << 24) | (self.document_index_view[index_offset+9] << 16) | (self.document_index_view[index_offset+10] << 8) | self.document_index_view[index_offset+11]
        return document_deserialize(self.documents_view[offset:offset+length])

    cpdef object _get_documents(self, list doc_ids, int limit=10):
        cdef list docs = []
        cdef object doc
        for doc_id in doc_ids:
            docs.append(self._get_document(doc_id))
            if len(docs) == limit:
                break
        return docs

    cpdef list _get_not_documents(self, list exclude_ids, int limit=10):
        cdef list docs = []
        cdef object doc
        cdef long long index_offset
        cdef long long offset
        cdef int length
        cdef bytes length_bytes
        cdef int docs_count = 0
        cdef int current_doc_id = 1

        while docs_count < limit:
            index_offset = (current_doc_id - 1) * 12
            offset = (<uint64_t>self.document_index_view[index_offset] << 56) | (<uint64_t>self.document_index_view[index_offset+1] << 48) | (<uint64_t>self.document_index_view[index_offset+2] << 40) | (<uint64_t>self.document_index_view[index_offset+3] << 32) | (<uint64_t>self.document_index_view[index_offset+4] << 24) | (<uint64_t>self.document_index_view[index_offset+5] << 16) | (<uint64_t>self.document_index_view[index_offset+6] << 8) | <uint64_t>self.document_index_view[index_offset+7]
            length = (self.document_index_view[index_offset+8] << 24) | (self.document_index_view[index_offset+9] << 16) | (self.document_index_view[index_offset+10] << 8) | self.document_index_view[index_offset+11]
            doc = document_deserialize(self.documents_view[offset:offset+length])
            if doc.id not in exclude_ids:
                docs.append(doc)
                docs_count += 1
            current_doc_id += 1
        return docs

    cpdef bint contains_phrase(self, object first_posting, object second_posting, int k=1):
        cdef int i=0, j=0
        while i < len(first_posting.positions) and j < len(second_posting.positions):
            if first_posting.positions[i] + k == second_posting.positions[j]:
                return True
            elif first_posting.positions[i] + k < second_posting.positions[j]:
                i += 1
            else:
                j += 1
        return False

    cpdef tuple evaluate_node(self, object node):
        if node is None:
            return PostingList(key=pst_id_key), False

        if node.left is None and node.right is None:
            if isinstance(node.value, list):
                result = self._get_postings(node.value[0])
                for token in node.value[1:]:
                    other_posting_list = self._get_postings(token, load_positions=True)
                    result.intersection(other_posting_list, self.contains_phrase).clone()
                return result, False
            else:
                return self._get_postings(node.value), False

        if node.value == "not":
            right_postings, right_is_not = self.evaluate_node(node.right)
            return right_postings, not right_is_not

        left_postings, left_is_not = self.evaluate_node(node.left)
        right_postings, right_is_not = self.evaluate_node(node.right)

        if node.value == "and":
            if not left_is_not and not right_is_not:
                return left_postings.intersection(right_postings), False
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
        else:
            raise ValueError(f"Unknown operator: {node.value}")

    cpdef list search(self, object query, int limit=10):
        if query.root is None:
            return []

        cdef list docs = []

        cdef clock_t start = clock()
        cdef object results
        cdef bint is_not
        results, is_not = self.evaluate_node(query.root)
        cdef clock_t end = clock()
        cdef double elapsed = (end - start) / CLOCKS_PER_SEC * 1000
        print(f"Query evaluation took {elapsed:.4f} milliseconds")

        start = clock()
        doc_ids = [res.doc_id for res in results]
        docs = self._get_not_documents(doc_ids, limit) if is_not else self._get_documents(doc_ids, limit)
        end = clock()
        elapsed = (end - start) / CLOCKS_PER_SEC * 1000
        print(f"Document retrieval took {elapsed:.4f} milliseconds")

        return docs
