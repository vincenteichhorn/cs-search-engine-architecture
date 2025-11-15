import mmap
import os
import struct
import time

from sea.posting_list import PostingList
from sea.document import Document
from sea.posting import Posting
from sea.util.gamma import BitReader
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t
from libc.stdint cimport uint8_t

def doc_id_key(doc):
    return doc.id

def pst_id_key(pst):
    return pst.doc_id


cdef class Engine:
    cdef public str index_path
    cdef object posting_lists_file
    cdef object posting_list_file_mmap
    cdef object posting_index_file
    cdef object document_index_file
    cdef object documents_file
    cdef dict token_dictionary

    def __init__(self, str index_path):
        self.index_path = index_path

        self.posting_lists_file = open(os.path.join(self.index_path, "posting_lists.bin"), "rb")
        self.posting_list_file_mmap = mmap.mmap(self.posting_lists_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.posting_index_file = open(os.path.join(self.index_path, "posting_lists_index.bin"), "rb")
        self.document_index_file = open(os.path.join(self.index_path, "document_index.bin"), "rb")
        self.documents_file = open(os.path.join(self.index_path, "documents.bin"), "rb")

        self.token_dictionary = {}
        cdef clock_t start = clock()
        self._load_token_dictionary()
        cdef clock_t end = clock()
        cdef double elapsed = (end - start) / CLOCKS_PER_SEC * 1000
        print(f"Loading token dictionary took {elapsed:.4f} milliseconds")

    cpdef void _load_token_dictionary(self):
        cdef bytes token_length_bytes, token_bytes, posting_list_offset_bytes, posting_list_length_bytes
        cdef int token_length, posting_list_length
        cdef long long posting_list_offset
        while True:
            token_length_bytes = self.posting_index_file.read(4)
            if not token_length_bytes:
                break
            token_length = struct.unpack(">I", token_length_bytes)[0]
            token_bytes = self.posting_index_file.read(token_length)
            posting_list_offset_bytes = self.posting_index_file.read(8)
            posting_list_offset = struct.unpack(">Q", posting_list_offset_bytes)[0]
            posting_list_length_bytes = self.posting_index_file.read(4)
            posting_list_length = struct.unpack(">I", posting_list_length_bytes)[0]
            self.token_dictionary[token_bytes.decode("utf-8")] = (posting_list_offset, posting_list_length)

    cpdef object _get_postings(self, str token, bint positions=False):
        if token not in self.token_dictionary:
            return PostingList(key=pst_id_key)

        cdef long long offset
        cdef int length
        offset, length = self.token_dictionary[token]
        cdef list postings = []
        cdef object posting
        cdef clock_t start = clock()

        cdef Py_ssize_t end_offset = offset + length
        cdef Py_ssize_t cur = offset
        cdef const uint8_t[:] buf = self.posting_list_file_mmap
        while cur < end_offset:
            posting, bytes_read = Posting.deserialize(buf[cur:end_offset], only_doc_id=not positions)
            cur += bytes_read
            postings.append(posting)

        cdef object posting_list = PostingList.from_list(postings, key=pst_id_key)

        cdef clock_t end = clock()
        cdef float duration = (end-start) / CLOCKS_PER_SEC * 1000
        print(f"Loading of posting list of token '{token}': {duration} milliseconds (length: {len(postings)})")

        return posting_list

    cpdef object _get_document(self, int doc_id):
        cdef long long index_offset = (doc_id - 1) * 12
        self.document_index_file.seek(index_offset)
        cdef bytes offset_bytes = self.document_index_file.read(8)
        cdef long long offset = struct.unpack(">Q", offset_bytes)[0]
        cdef bytes length_bytes = self.document_index_file.read(4)
        cdef int length = struct.unpack(">I", length_bytes)[0]
        self.documents_file.seek(offset)
        cdef bytes doc_bytes = self.documents_file.read(length)
        return Document.deserialize(bytearray(doc_bytes))

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
        cdef object idx_file = self.document_index_file
        cdef object docs_file = self.documents_file
        cdef bytes offset_bytes
        cdef bytes length_bytes
        cdef bytes doc_bytes
        cdef long long offset
        cdef int length
        cdef int docs_count = 0

        while docs_count < limit:
            offset_bytes = idx_file.read(8)
            if not offset_bytes:
                break
            offset = struct.unpack(">Q", offset_bytes)[0]

            length_bytes = idx_file.read(4)
            if not length_bytes:
                break
            length = struct.unpack(">I", length_bytes)[0]

            docs_file.seek(offset)
            doc_bytes = docs_file.read(length)
            doc = Document.deserialize(bytearray(doc_bytes))
            if doc.id not in exclude_ids:
                docs.append(doc)
                docs_count += 1
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
                    other_posting_list = self._get_postings(token, positions=True)
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
