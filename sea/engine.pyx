from sea.corpus cimport Corpus, doc_to_dict
from sea.tokenizer cimport Tokenizer, TokenizedField
from sea.document cimport Document, Posting, deserialize_postings, free_posting
from sea.posting_list cimport intersection, union, difference
from sea.query cimport QueryParser, QueryNode, print_query_tree
from sea.util.disk_array cimport DiskArray, EntryInfo, DiskArrayIterator
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
import os
from libc.stdlib cimport malloc
from libcpp.utility cimport pair
from sea.spelling_corrector cimport SpellingCorrector
from libcpp.unordered_map cimport unordered_map

cdef str TIER_PREFIX = "tier_"
cdef size_t SNIPPET_RADIUS = 150
cdef uint32_t SPELLING_FREQUENCY_THRESHOLD = 100

ctypedef char* CharPtr

cdef class Engine:

    cdef str save_path
    cdef Corpus corpus
    cdef Tokenizer tokenizer
    cdef QueryParser query_parser
    cdef SpellingCorrector spelling_corrector

    cdef unordered_map[uint64_t, uint32_t] global_document_frequencies

    cdef dict tier_disk_arrays
    cdef uint64_t and_operator
    cdef uint64_t or_operator
    cdef uint64_t not_operator
    cdef vector[Posting] free_me_pls

    def __cinit__(self, save_path):

        self.save_path = save_path
        self.corpus = Corpus(save_path, "", mmap=True)
        self.tokenizer = Tokenizer(save_path, mmap=True)

        self.tier_disk_arrays = {}
        for path in os.listdir(save_path):
            if path.startswith(TIER_PREFIX):
                try:
                    tier_id = int(path[len(TIER_PREFIX):])
                except ValueError:
                    continue
                self.tier_disk_arrays[tier_id] = DiskArray(os.path.join(self.save_path, path))
        
        self.global_document_frequencies = unordered_map[uint64_t, uint32_t]()
        cdef DiskArrayIterator it = self.tier_disk_arrays[0].iterator()
        cdef EntryInfo entry
        cdef uint64_t token_id = 0
        while it.has_next():
            entry = it.next_entry()
            self.global_document_frequencies[token_id] = entry.payload
            token_id += 1
        
        self.tokenizer.py_tokenize(b"and or not ( ) \"", True)  # Preload special tokens into vocabulary
        self.and_operator=self.tokenizer.py_vocab_lookup(b"and")
        self.or_operator=self.tokenizer.py_vocab_lookup(b"or")
        self.not_operator=self.tokenizer.py_vocab_lookup(b"not")
        self.query_parser = QueryParser(
            and_operator=self.and_operator,
            or_operator=self.or_operator,
            not_operator=self.not_operator,
            open_paren=self.tokenizer.py_vocab_lookup(b"("),
            close_paren=self.tokenizer.py_vocab_lookup(b")"),
            phrase_marker=self.tokenizer.py_vocab_lookup(b"\"")
        )
        self.spelling_corrector = SpellingCorrector(self.tokenizer, self.global_document_frequencies, exclude_threshold=SPELLING_FREQUENCY_THRESHOLD)
        self.free_me_pls = vector[Posting]()

    cdef vector[Posting] _get_postings(self, uint64_t token_id, uint32_t tier):
        
        cdef DiskArray tier_disk_array
        tier_disk_array = self.tier_disk_arrays[tier]
        cdef EntryInfo entry = tier_disk_array.get(token_id)
        cdef vector[Posting] postings = deserialize_postings(entry.data, entry.length)
        self.free_me_pls.insert(self.free_me_pls.end(), postings.begin(), postings.end())
        return postings
    
    cdef pair[CharPtr, uint32_t] _get_snippet(self, Document doc, Posting posting):

        cdef size_t snippet_length = SNIPPET_RADIUS
        cdef uint32_t position = posting.char_positions[0] - doc.title_length if posting.char_positions.size() > 0 else 0
        if position >= doc.body_length:
            position = doc.body_length // 2
        cdef size_t start_pos = position - snippet_length // 2 if position >= snippet_length // 2 else 0
        cdef size_t end_pos = start_pos + snippet_length if start_pos + snippet_length < doc.body_length else doc.body_length

        cdef char* snippet = <char*>malloc((end_pos - start_pos + 1) * sizeof(char))
        cdef size_t i
        cdef uint32_t first_space = 0, last_space = 0

        for i in range(start_pos, end_pos):
            if doc.body[i] == ' ':
                first_space = i
                break

        for i in range(end_pos, start_pos, -1):
            if doc.body[i] == ' ':
                last_space = i
                break

        if first_space > 0:
            start_pos = first_space + 1
        if last_space > 0 and last_space > (start_pos - start_pos):
            end_pos = last_space

        for i in range(start_pos, end_pos):
            snippet[i - start_pos] = doc.body[i]
        snippet[end_pos - start_pos] = '\0'
        return pair[CharPtr, uint32_t](snippet, end_pos - start_pos)
    
    cdef list _retrieve_documents(self, vector[Posting] postings, uint32_t top_k=10):

        cdef list documents = []
        cdef Document doc

        for i in range(postings.size()):
            doc = self.corpus.get_document(postings[i].doc_id, lowercase=False)
            doc.score = postings[i].score
            snippet_pair = self._get_snippet(doc, postings[i])
            doc.snippet = snippet_pair.first
            doc.snippet_length = snippet_pair.second
            documents.append(doc_to_dict(doc))
            if len(documents) >= top_k:
                break
        documents = sorted(documents, key=lambda x: x['score'], reverse=True)
        return documents
    
    cdef pair[vector[Posting], bint] _full_boolean_search(self, QueryNode* node, uint32_t tier):


        if node == NULL:
            return pair[vector[Posting], bint](vector[Posting](), False)

        cdef vector[Posting] left_postings, right_postings, postings
        cdef pair[vector[Posting], bint] left_pair, right_pair, tmp_pair
        cdef uint64_t token, i
    
        if node.left == NULL and node.right == NULL:
            if node.values.size() > 1:
                result = self._get_postings(node.values[0], tier)
                for i in range(1, node.values.size()):
                    token = node.values[i]
                    other_posting_list = self._get_postings(token, tier)
                    intersection(result, other_posting_list, True)
                return pair[vector[Posting], bint](result, False)
            else:
                return pair[vector[Posting], bint](self._get_postings(node.values[0], tier), False)

        if node.values[0] == self.not_operator:
            tmp_pair = self._full_boolean_search(node.right, tier)
            tmp_pair.second = not tmp_pair.second
            return tmp_pair

        left_pair = self._full_boolean_search(node.left, tier)
        right_pair = self._full_boolean_search(node.right, tier)

        if node.values[0] == self.and_operator:
            if not left_pair.second and not right_pair.second:
                intersection(left_pair.first, right_pair.first, False)
                return pair[vector[Posting], bint](left_pair.first, False)
            elif left_pair.second and not right_pair.second:
                difference(right_pair.first, left_pair.first)
                return pair[vector[Posting], bint](left_pair.first, False)
            elif not left_pair.second and right_pair.second:
                difference(left_pair.first, right_pair.first)
                return pair[vector[Posting], bint](left_pair.first, False)
            else:
                union(left_pair.first, right_pair.first)
                return pair[vector[Posting], bint](left_pair.first, False)
        elif node.values[0] == self.or_operator:
            if not left_pair.second and not right_pair.second:
                union(left_pair.first, right_pair.first)
                return pair[vector[Posting], bint](left_pair.first, False)
            elif left_pair.second and not right_pair.second:
                difference(left_pair.first, right_pair.first)
                return pair[vector[Posting], bint](left_pair.first, False)
            elif not left_pair.second and right_pair.second:
                difference(right_pair.first, left_pair.first)
                return pair[vector[Posting], bint](left_pair.first, False)
            else:
                intersection(left_pair.first, right_pair.first, False)
                return pair[vector[Posting], bint](left_pair.first, False)
        else:
            return pair[vector[Posting], bint](vector[Posting](), False)
    
    cdef void free_all_postings(self):

        cdef uint32_t i
        for i in range(self.free_me_pls.size()):
            free_posting(&self.free_me_pls[i], True)
        self.free_me_pls.clear()
        
    cpdef list search(self, str query, size_t top_k):

        cdef bytes query_bytes = query.encode("utf-8")
        cdef const char* query_c = query_bytes
        cdef TokenizedField tokenized_query = self.tokenizer.tokenize(query_c, len(query_bytes), True)

        cdef pair[CharPtr, uint32_t] query_correction = self.spelling_corrector.get_top_correction(tokenized_query.tokens, min_similarity=0.75)
        if query_correction.second > 0:
            print("Did you mean?", str(query_correction.first, "utf-8"))

        cdef QueryNode* query_tree = self.query_parser.parse(tokenized_query.tokens)
        # print("Query Tree: ")
        # print_query_tree(query_tree, 0)

        cdef pair[vector[Posting], bint] result_pair = self._full_boolean_search(query_tree, tier=4)

        cdef list results = self._retrieve_documents(result_pair.first, top_k)

        self.free_all_postings()

        return results