import mmap
import os
import struct
import time

from sea.posting_list import PostingList
from sea.document import Document
from sea.posting import Posting
from sea.util.gamma import BitReader
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t


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
        self._load_token_dictionary()

    cpdef void _load_token_dictionary(self):
        cdef bytes token_length_bytes, token_bytes, posting_list_offset_bytes, posting_list_length_bytes
        cdef int token_length, posting_list_offset, posting_list_length
        while True:
            token_length_bytes = self.posting_index_file.read(4)
            if not token_length_bytes:
                break
            token_length = struct.unpack(">I", token_length_bytes)[0]
            token_bytes = self.posting_index_file.read(token_length)
            posting_list_offset_bytes = self.posting_index_file.read(4)
            posting_list_offset = struct.unpack(">I", posting_list_offset_bytes)[0]
            posting_list_length_bytes = self.posting_index_file.read(4)
            posting_list_length = struct.unpack(">I", posting_list_length_bytes)[0]
            self.token_dictionary[token_bytes.decode("utf-8")] = (posting_list_offset, posting_list_length)

    cpdef object _get_postings(self, str token, bint positions=False):
        if token not in self.token_dictionary:
            return PostingList(key=pst_id_key)

        cdef int offset, length
        offset, length = self.token_dictionary[token]
        posting_list_bytes = self.posting_list_file_mmap[offset: offset + length]
        print(f"Reading postings for token '{token}' from offset {offset} with length {length}")
        cdef list postings = []
        cdef object posting
        cdef clock_t start = clock()
        remainder = posting_list_bytes
        while len(remainder) > 0:
            posting, remainder = Posting.deserialize(remainder, only_doc_id=not positions)
            postings.append(posting)
        cdef clock_t end = clock()
        cdef double duration = (end - start) / CLOCKS_PER_SEC
        print(f"Deserialized postings for token '{token}' in {duration*1000:.8f} milliseconds")

        return PostingList.from_list(postings, key=pst_id_key)

    cpdef object _get_document(self, int doc_id):
        cdef int index_offset = (doc_id - 1) * 8
        self.document_index_file.seek(index_offset)
        cdef bytes offset_bytes = self.document_index_file.read(4)
        cdef int offset = struct.unpack(">I", offset_bytes)[0]
        cdef bytes length_bytes = self.document_index_file.read(4)
        cdef int length = struct.unpack(">I", length_bytes)[0]
        self.documents_file.seek(offset)
        cdef bytes doc_bytes = self.documents_file.read(length)
        return Document.deserialize(bytearray(doc_bytes))

    cpdef list _get_not_documents(self, list exclude_ids, int limit=10):
        cdef int index_offset = 0
        cdef list docs = []
        cdef object doc
        cdef int offset, length
        while len(docs) < limit:
            self.document_index_file.seek(index_offset)
            offset_bytes = self.document_index_file.read(4)
            offset = struct.unpack(">I", offset_bytes)[0]
            length_bytes = self.document_index_file.read(4)
            length = struct.unpack(">I", length_bytes)[0]
            self.documents_file.seek(offset)
            doc_bytes = self.documents_file.read(length)
            doc = Document.deserialize(bytearray(doc_bytes))
            if doc.id not in exclude_ids:
                docs.append(doc)
            index_offset += 8
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
                result = self._get_postings(node.value[0]).clone()
                for token in node.value[1:]:
                    other_posting_list = self._get_postings(token, positions=True)
                    result.intersection(other_posting_list, self.contains_phrase).clone()
                return result, False
            else:
                return self._get_postings(node.value).clone(), False

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

        results, is_not = self.evaluate_node(query.root)

        cdef clock_t start = clock()
        if is_not:
            docs = self._get_not_documents([res.doc_id for res in results], limit)

        for res in results:
            if len(docs) == limit:
                break
            docs.append(self._get_document(res.doc_id))
        cdef clock_t end = clock()
        cdef double duration = (end - start) / CLOCKS_PER_SEC
        print(f"Doc retreival completed in {duration*1000:.8f} milliseconds")

        return docs
