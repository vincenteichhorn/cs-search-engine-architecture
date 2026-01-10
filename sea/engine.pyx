from sea.corpus cimport Corpus, doc_to_dict
from sea.tokenizer cimport Tokenizer, TokenizedField
from sea.document cimport Document, Posting, TokenizedDocument, deserialize_postings, free_posting, free_tokenized_document_with_postings, get_posting_list_length
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
import json
from sea.learning_to_rank.model import ListNet
import torch
from libcpp.algorithm cimport sort
from sea.learning_to_rank.feature_mapping cimport get_features
from cpython.unicode cimport PyUnicode_DecodeUTF8
from cython.cimports.libc.stdint cimport UINT32_MAX

cdef str TIER_PREFIX = "tier_"
cdef size_t SNIPPET_RADIUS = 150
cdef uint32_t SPELLING_FREQUENCY_THRESHOLD = 100
cdef uint32_t NUM_TOTAL_DOCS = 3_213_835
cdef list BM25_FIELD_BOOSTS = [1.0, 0.5]
cdef list BM25_BS = [0.75, 0.75]
cdef float BM25_K = 1.5
cdef list AVG_FIELD_LENGTHS = [4.358767951683892, 783.4649271042229]

ctypedef char* CharPtr

cdef bint compare_postings_scores(Posting a, Posting b) noexcept nogil:
    return a.score > b.score

cdef class Engine:

    cdef str save_path
    cdef Corpus corpus
    cdef Tokenizer tokenizer
    cdef QueryParser query_parser
    cdef SpellingCorrector spelling_corrector

    cdef unordered_map[uint64_t, uint64_t] global_document_frequencies

    cdef dict tier_disk_arrays
    cdef uint32_t num_tiers
    cdef uint64_t and_operator
    cdef uint64_t or_operator
    cdef uint64_t not_operator

    cdef uint64_t num_total_docs
    cdef vector[float] average_field_lengths
    cdef float bm25_k
    cdef vector[float] bm25_bs
    cdef object model


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
        self.num_tiers = len(self.tier_disk_arrays)
        
        self.global_document_frequencies = unordered_map[uint64_t, uint64_t]()
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

        self.num_total_docs = NUM_TOTAL_DOCS
        self.average_field_lengths = vector[float]()
        for avg_len in AVG_FIELD_LENGTHS:
            self.average_field_lengths.push_back(avg_len)
        self.bm25_k = BM25_K
        self.bm25_bs = vector[float]()
        for b in BM25_BS:
            self.bm25_bs.push_back(b)

        cdef dict config
        with open("./models/config.json", "r") as f:
            config = json.load(f)
        self.model = ListNet(**config)
        self.model.load_state_dict(torch.load("./models/listnet_latest.pth"))
        self.model.eval()
    
    cdef uint32_t _tier_min_posting_list_length(self, QueryNode* node, uint32_t tier):
        
        cdef uint32_t min_length
        cdef uint64_t token
        cdef DiskArray tier_disk_array
        cdef EntryInfo entry
        cdef uint32_t posting_list_length

        if node == NULL:
            return UINT32_MAX
        if node.left == NULL and node.right == NULL:
            min_length = UINT32_MAX
            tier_disk_array = self.tier_disk_arrays[tier]
            for i in range(node.values.size()):
                token = node.values[i]
                entry = tier_disk_array.get(token)
                posting_list_length = get_posting_list_length(entry.data, entry.length)
                if posting_list_length < min_length:
                    min_length = posting_list_length
            return min_length
        return min(self._tier_min_posting_list_length(node.left, tier), self._tier_min_posting_list_length(node.right, tier))

    cdef uint32_t _min_tier_index(self, QueryNode* node, uint32_t top_k):
        cdef uint32_t tier
        cdef uint32_t min_length
        for tier in range(self.num_tiers):
            min_length = self._tier_min_posting_list_length(node, tier)
            if min_length >= top_k:
                return tier
        return self.num_tiers - 1

    cdef vector[Posting] _get_postings(self, uint64_t token_id, uint32_t min_tier, uint32_t max_tier):
        
        cdef DiskArray tier_disk_array
        cdef EntryInfo entry
        cdef vector[Posting] postings, tmp_postings
        for tier in range(min_tier, max_tier + 1):
            tier_disk_array = self.tier_disk_arrays[tier]
            entry = tier_disk_array.get(token_id)
            tmp_postings = deserialize_postings(entry.data, entry.length)
            postings.insert(postings.end(), tmp_postings.begin(), tmp_postings.end())
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
    
    cdef vector[TokenizedDocument] _retrieve_tokenized_documents(self, vector[Posting] postings, uint32_t top_k) noexcept nogil:

        cdef vector[TokenizedDocument] tokenized_documents = vector[TokenizedDocument]()
        cdef TokenizedDocument doc
        for i in range(postings.size()):
            with gil:
                doc = self.corpus.get_tokenized_document(postings[i].doc_id, tokenizer=self.tokenizer)
            tokenized_documents.push_back(doc)
            if tokenized_documents.size() >= top_k:
                break
        return tokenized_documents

    cdef vector[Document] _retrieve_documents_with_snippets(self, vector[Posting]& postings, vector[uint32_t] indices, uint32_t top_k) noexcept nogil:

        cdef vector[Document] documents = vector[Document]()
        cdef Document doc
        cdef pair[CharPtr, uint32_t] snippet_pair
        cdef size_t i
        if indices.size() <= 0:
            indices = vector[uint32_t]()
            for i in range(postings.size()):
                indices.push_back(i)

        for i in range(indices.size()):
            doc = self.corpus.get_document(postings[indices[i]].doc_id, lowercase=False)
            doc.score = postings[indices[i]].score
            with gil:
                snippet_pair = self._get_snippet(doc, postings[indices[i]])
            doc.snippet = snippet_pair.first
            doc.snippet_length = snippet_pair.second
            documents.push_back(doc)
            if documents.size() >= top_k:
                break
        return documents
    
    cdef pair[vector[Posting], bint] _full_boolean_search(self, QueryNode* node, uint32_t min_tier, uint32_t max_tier):


        if node == NULL:
            return pair[vector[Posting], bint](vector[Posting](), False)

        cdef vector[Posting] left_postings, right_postings, postings
        cdef pair[vector[Posting], bint] left_pair, right_pair, tmp_pair
        cdef uint64_t token, i
    
        if node.left == NULL and node.right == NULL:
            if node.values.size() > 1:
                result = self._get_postings(node.values[0], min_tier, max_tier)
                for i in range(1, node.values.size()):
                    token = node.values[i]
                    other_posting_list = self._get_postings(token, min_tier, max_tier)
                    intersection(result, other_posting_list, True)
                return pair[vector[Posting], bint](result, False)
            else:
                return pair[vector[Posting], bint](self._get_postings(node.values[0], min_tier, max_tier), False)

        if node.values[0] == self.not_operator:
            tmp_pair = self._full_boolean_search(node.right, min_tier, max_tier)
            tmp_pair.second = not tmp_pair.second
            return tmp_pair

        left_pair = self._full_boolean_search(node.left, min_tier, max_tier)
        right_pair = self._full_boolean_search(node.right, min_tier, max_tier)

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
    
    cdef vector[uint32_t] _rank_documents(self, vector[uint64_t] query_tokens, vector[TokenizedDocument]& tokenized_documents):
        cdef float[:, :] feature_matrix = get_features(
            query_tokens, 
            tokenized_documents, 
            self.global_document_frequencies, 
            self.num_total_docs, 
            self.average_field_lengths, 
            self.bm25_k, 
            self.bm25_bs
        )
        cdef object input = torch.tensor([feature_matrix])
        cdef object output = self.model(input)[0]
        cdef list ranked_document_indices = list(torch.argsort(output, descending=True))
        cdef vector[uint32_t] ranked_documents = vector[uint32_t]()
        for i in range(len(ranked_document_indices)):
            ranked_documents.push_back(ranked_document_indices[i])
        return ranked_documents
        
    cpdef list search(self, str query, size_t top_k):

        cdef bytes query_bytes = query.lower().encode("utf-8")
        cdef const char* query_c = query_bytes
        cdef TokenizedField tokenized_query = self.tokenizer.tokenize(query_c, len(query_bytes), True)

        cdef pair[CharPtr, uint32_t] query_correction = self.spelling_corrector.get_top_correction(tokenized_query.tokens, min_similarity=0.75)
        if query_correction.second > 0:
            print(f"Did you mean \033[4m{str(query_correction.first, 'utf-8')}\033[0m?")

        cdef QueryNode* query_tree = self.query_parser.parse(tokenized_query.tokens)

        cdef uint32_t tier = self._min_tier_index(query_tree, top_k)

        cdef list results = []
        cdef pair[vector[Posting], bint] results_pair = self._full_boolean_search(query_tree, min_tier=0, max_tier=tier)
        if results_pair.first.size() <= 0:
            return results

        sort(results_pair.first.begin(), results_pair.first.end(), compare_postings_scores)
        cdef vector[TokenizedDocument] tokenized_documents = self._retrieve_tokenized_documents(results_pair.first, top_k*3)
        cdef vector[uint32_t] ranking_indices = self._rank_documents(tokenized_query.tokens, tokenized_documents)

        cdef vector[Document] documents = self._retrieve_documents_with_snippets(results_pair.first, ranking_indices, top_k)
        results = [doc_to_dict(documents[i]) for i in range(top_k)]

        for i in range(results_pair.first.size()):
            free_posting(&results_pair.first[i], True)
        # for i in range(tokenized_documents.size()):
        #     free_tokenized_document_with_postings(&tokenized_documents[i])

        return results