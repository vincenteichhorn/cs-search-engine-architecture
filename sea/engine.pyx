import mmap
import os

from sea.posting_list import PostingList, posting_list_from_list
from sea.document import Document, document_deserialize
from sea.posting import Posting
from sea.util.gamma import BitReader
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t
from libc.stdint cimport uint8_t, uint64_t
from cpython.unicode cimport PyUnicode_DecodeUTF8
from sea.posting import posting_deserialize
from sea.spelling_corrector import SpellingCorrector

cpdef doc_id_key(doc):
    return doc.id

cpdef pst_id_key(pst):
    return pst.doc_id

cpdef first_of_tuple(object tpl):
    return tpl[0]


cdef class Engine:
    cdef object config
    cdef list posting_list_files
    cdef list posting_list_mmaps
    cdef list posting_list_views
    cdef list posting_list_ptrs

    cdef object document_index_file
    cdef object document_index_mmap
    cdef const uint8_t[:] document_index_view
    cdef const uint8_t* document_index_ptr

    cdef object documents_file
    cdef object documents_mmap
    cdef const uint8_t[:] documents_view
    cdef const uint8_t* documents_ptr


    cdef list token_dictionaries
    cdef set frequent_terms
    cdef object spelling_corrector

    def __init__(self, object config):
        self.config = config
        self.posting_list_files = []
        self.posting_list_mmaps = []
        self.posting_list_views = []
        self.posting_list_ptrs = []
        self.initialize()

    cpdef initialize(self):

        cdef const uint8_t[:] tmp_view
        cdef const uint8_t* tmp_ptr
        cdef object tmp_file
        cdef object tmp_mmap 
        cdef str file_path
        cdef int file_size
        for tier in range(self.config.NUM_TIERS):
            file_path = os.path.join(self.config.INDEX_PATH, f"{self.config.TIER_PREFIX}{tier}", self.config.POSTINGS_DATA_FILE_NAME)
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                tmp_file = open(file_path, "rb")
                tmp_mmap = mmap.mmap(tmp_file.fileno(), 0, access=mmap.ACCESS_READ)
                tmp_view = tmp_mmap
                tmp_ptr = (&tmp_view[0])
                self.posting_list_files.append(tmp_file)
                self.posting_list_mmaps.append(tmp_mmap)
                self.posting_list_views.append(tmp_view)
                self.posting_list_ptrs.append(tmp_ptr)
            else:
                self.posting_list_files.append(None)
                self.posting_list_mmaps.append(None)
                self.posting_list_views.append(None)
                self.posting_list_ptrs.append(None)

        self.document_index_file = open(os.path.join(self.config.INDEX_PATH, self.config.DOCUMENTS_INDEX_FILE_NAME), "rb")
        self.document_index_mmap = mmap.mmap(self.document_index_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.document_index_view = self.document_index_mmap
        self.document_index_ptr = &self.document_index_view[0]

        self.documents_file = open(os.path.join(self.config.INDEX_PATH, self.config.DOCUMENTS_DATA_FILE_NAME), "rb")
        self.documents_mmap = mmap.mmap(self.documents_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.documents_view = self.documents_mmap
        self.documents_ptr = &self.documents_view[0]

        self.token_dictionaries = []
        self.frequent_terms = set()
        cdef clock_t start = clock()
        self._load_token_dictionary()
        cdef clock_t end = clock()
        cdef double elapsed = (end - start) / CLOCKS_PER_SEC * 1000
        print(f"- Loading token dictionary took {elapsed:.4f} milliseconds")

        start = clock()
        self.spelling_corrector = SpellingCorrector(list(self.frequent_terms))
        end = clock()
        elapsed = (end - start) / CLOCKS_PER_SEC * 1000
        print(f"- Loading spelling corrector took {elapsed:.4f} milliseconds")

    cpdef void _load_token_dictionary(self):

        cdef const uint8_t[:] tmp_view
        cdef const uint8_t* tmp_ptr
        cdef object tmp_file
        cdef object tmp_mmap 
        cdef unsigned int file_size = 0
        cdef unsigned int pos
        cdef str token
        cdef unsigned long long posting_list_offset
        cdef unsigned int posting_list_length
        cdef str file_path
        for tier in range(self.config.NUM_TIERS):
            self.token_dictionaries.append({})
            file_path = os.path.join(self.config.INDEX_PATH, f"{self.config.TIER_PREFIX}{tier}", self.config.POSTINGS_INDEX_FILE_NAME)
            tmp_file = open(file_path, "rb")
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                continue
            tmp_mmap = mmap.mmap(tmp_file.fileno(), 0, access=mmap.ACCESS_READ)
            tmp_view = tmp_mmap
            tmp_ptr = &tmp_view[0]
            file_size = tmp_view.shape[0]
            pos = 0
            while pos < file_size:
                token_length = (tmp_view[pos] << 24) | (tmp_view[pos + 1] << 16) | (tmp_view[pos + 2] << 8) | tmp_view[pos + 3]
                pos += 4
                token = PyUnicode_DecodeUTF8(<char*>(tmp_ptr + pos), token_length, NULL)
                pos += token_length
                posting_list_offset = (<uint64_t>tmp_view[pos] << 56) | (<uint64_t>tmp_view[pos + 1] << 48) | (<uint64_t>tmp_view[pos + 2] << 40) | (<uint64_t>tmp_view[pos + 3] << 32) | (<uint64_t>tmp_view[pos + 4] << 24) | (<uint64_t>tmp_view[pos + 5] << 16) | (<uint64_t>tmp_view[pos + 6] << 8) | <uint64_t>tmp_view[pos + 7]
                pos += 8
                posting_list_length = (tmp_view[pos] << 24) | (tmp_view[pos + 1] << 16) | (tmp_view[pos + 2] << 8) | tmp_view[pos + 3]
                pos += 4
                term_doc_freq = (tmp_view[pos] << 24) | (tmp_view[pos + 1] << 16) | (tmp_view[pos + 2] << 8) | tmp_view[pos + 3]
                pos += 4
                self.token_dictionaries[tier][token] = (posting_list_offset, posting_list_length, term_doc_freq)
                if term_doc_freq >= self.config.SPELLING_FREQUENCY_THRESHOLD:
                    self.frequent_terms.add(token)

    cpdef object _get_postings(self, str token, int tier):
        if token not in self.token_dictionaries[tier]:
            return PostingList(key=pst_id_key)

        cdef long long offset
        cdef int length
        offset, length, _ = self.token_dictionaries[tier][token]
        cdef list postings = []
        cdef object posting
        cdef Py_ssize_t bytes_read
        cdef Py_ssize_t end_offset = offset + length
        cdef Py_ssize_t cur = offset
        while cur < end_offset:
            posting, bytes_read = posting_deserialize(self.posting_list_views[tier][cur:end_offset])
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

    cpdef bint contains_phrase(self, object first_posting, object second_posting, int k=-1):
        cdef int i=0, j=0
        if k == -1:
            k = self.config.IN_PHRASE_CHARACTER_DISTANCE
        while i < len(first_posting.char_positions) and j < len(second_posting.char_positions):
            if first_posting.char_positions[i] + k == second_posting.char_positions[j]:
                return True
            elif first_posting.char_positions[i] + k < second_posting.char_positions[j]:
                i += 1
            else:
                j += 1
        return False

    cpdef object add_bm25(self, object first_posting, object second_posting):
        second_posting.score += first_posting.score
        return second_posting

    cpdef tuple evaluate_node(self, object node, int tier):
        if node is None:
            return PostingList(key=pst_id_key), False

        if node.left is None and node.right is None:
            if isinstance(node.value, list):
                result = self._get_postings(node.value[0], tier)
                for token in node.value[1:]:
                    other_posting_list = self._get_postings(token, tier)
                    result.intersection(other_posting_list, self.contains_phrase, self.add_bm25)
                return result, False
            else:
                return self._get_postings(node.value, tier), False

        if node.value == "not":
            right_postings, right_is_not = self.evaluate_node(node.right, tier)
            return right_postings, not right_is_not

        left_postings, left_is_not = self.evaluate_node(node.left, tier)
        right_postings, right_is_not = self.evaluate_node(node.right, tier)

        if node.value == "and":
            if not left_is_not and not right_is_not:
                return left_postings.intersection(right_postings, merge_items=self.add_bm25), False
            elif left_is_not and not right_is_not:
                return right_postings.difference(left_postings), False
            elif not left_is_not and right_is_not:
                return left_postings.difference(right_postings), False
            else:
                return left_postings.union(right_postings, merge_items=self.add_bm25), True
        elif node.value == "or":
            if not left_is_not and not right_is_not:
                return left_postings.union(right_postings, merge_items=self.add_bm25), False
            elif left_is_not and not right_is_not:
                return left_postings.difference(right_postings), True
            elif not left_is_not and right_is_not:
                return right_postings.difference(left_postings), True
            else:
                return left_postings.intersection(right_postings, merge_items=self.add_bm25), True
        else:
            raise ValueError(f"Unknown operator: {node.value}")
    
    cdef str get_document_snippet(self, object document, object posting):
        cdef int first_pos = posting.char_positions[0]
        cdef int start_pos = max(0, first_pos - self.config.SNIPPET_RADIUS)
        cdef int end_pos = min(len(document.body), first_pos + self.config.SNIPPET_RADIUS)
        cdef str snippet = document.body[start_pos:end_pos]
        cdef int leading_space = snippet.find(' ')
        cdef int trailing_space = snippet.rfind(' ')
        if leading_space != -1:
            snippet = snippet[leading_space+1:]
        if trailing_space != -1:
            snippet = snippet[:trailing_space]
        return snippet

    cpdef list search(self, object query, int limit=10):
        
        if query.root is None:
            return []

        cdef list corrected_query_tokens = query.tokens
        for i, qtok in enumerate(query.tokens):
            corrected_query_tokens[i] = self.spelling_corrector.get_top_correction(qtok)
        cdef str corrected_query = " ".join(corrected_query_tokens)
        if corrected_query != query.input:
            print(f"Did you mean: {corrected_query}?")
        
        cdef object postings = PostingList(key=pst_id_key)
        cdef object tmp_posting_list
        cdef bint is_not = False
        cdef int tier = 0

        cdef clock_t start = clock()

        while len(postings) < limit and tier < self.config.NUM_TIERS:

            if self.posting_list_files[tier] is None:
                tier += 1
                continue
            tmp_posting_list, is_not = self.evaluate_node(query.root, tier)
            postings.union(tmp_posting_list, merge_items=self.add_bm25)
            tier += 1

        cdef clock_t end = clock()
        cdef double elapsed = (end - start) / CLOCKS_PER_SEC * 1000
        print(f"- Query evaluation took {elapsed:.4f} milliseconds")

        cdef list docs
        cdef list doc_ids
        cdef list scores
        doc_ids = [res.doc_id for res in postings]
        scores = [res.score if not is_not else 0.0 for res in postings]
        docs = self._get_not_documents(doc_ids, limit) if is_not else self._get_documents(doc_ids, limit)
        cdef list snippets = []
        for doc, posting in zip(docs, postings):
            snippets.append(self.get_document_snippet(doc, posting))

        search_results = sorted(zip(scores, docs, snippets), key=first_of_tuple, reverse=True)

        return search_results
