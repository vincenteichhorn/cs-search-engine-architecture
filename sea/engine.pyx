import os
import json
import torch
import numpy as np
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from libc.stdlib cimport malloc
from libcpp.utility cimport pair
from sea.spelling_corrector cimport SpellingCorrector
from libcpp.unordered_map cimport unordered_map
from libcpp.algorithm cimport sort
from cython.cimports.libc.stdint cimport UINT32_MAX
from libcpp.unordered_set cimport unordered_set
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC


from sea.corpus cimport Corpus, doc_to_dict
from sea.tokenizer cimport Tokenizer, TokenizedField
from sea.document cimport Document, Posting, TokenizedDocument, SearchResultPosting, deserialize_search_result_postings, free_posting, free_tokenized_document, get_posting_list_length
from sea.posting_list cimport intersection, intersection_phrase, union, difference
from sea.query cimport QueryParser, QueryNode, print_query_tree
from sea.util.disk_array cimport DiskArray, EntryInfo, DiskArrayIterator
from sea.learning_to_rank.model import ListNet
from sea.learning_to_rank.feature_mapping cimport get_features

cdef str TIER_PREFIX = "tier_"
cdef size_t SNIPPET_RADIUS = 100
cdef uint32_t SPELLING_FREQUENCY_THRESHOLD = 100
cdef uint32_t NUM_TOTAL_DOCS = 3_213_835
cdef list BM25_FIELD_BOOSTS = [1.0, 0.5]
cdef list BM25_BS = [0.75, 0.75]
cdef float BM25_K = 1.5
cdef list AVG_FIELD_LENGTHS = [4.358767951683892, 783.4649271042229]
cdef int MAT_DIM = 64

ctypedef char* CharPtr

cdef bint compare_postings_scores(SearchResultPosting& a, SearchResultPosting& b) noexcept nogil:
    return a.total_score > b.total_score

cdef bint compare_postings_doc_ids(SearchResultPosting& a, SearchResultPosting& b) noexcept nogil:
    return a.doc_id < b.doc_id

cdef class Engine:

    cdef str name
    cdef str index_path
    cdef str embeddings_path
    cdef str model_path
    cdef public Corpus corpus
    cdef public Tokenizer tokenizer
    cdef QueryParser query_parser
    cdef SpellingCorrector spelling_corrector

    cdef unordered_map[uint64_t, uint64_t] global_document_frequencies

    # lookup order: token_id -> min_tier -> max_tier -> vector[SearchResultPosting]
    cdef unordered_map[uint64_t, unordered_map[uint32_t, unordered_map[uint32_t, unordered_map[bint, vector[SearchResultPosting]]]]] postings_cache

    cdef dict tier_disk_arrays
    cdef uint32_t num_tiers
    cdef uint64_t and_operator
    cdef uint64_t or_operator
    cdef uint64_t not_operator

    cdef uint64_t num_total_docs
    cdef vector[float] average_field_lengths
    cdef float bm25_k
    cdef vector[float] bm25_bs
    cdef vector[float] bm25_field_boosts

    cdef str device
    cdef object ltr_model

    cdef object corpus_embeddings
    cdef object embed_model
    cdef object embed_tokenizer


    def __cinit__(self, name, index_path, embeddings_path, model_path):

        self.name = name
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.model_path = model_path
        self.corpus = Corpus(os.path.join(self.index_path, self.name), "", mmap=True)
        self.tokenizer = Tokenizer(os.path.join(self.index_path, self.name), mmap=True)

        self.postings_cache = unordered_map[uint64_t, unordered_map[uint32_t, unordered_map[uint32_t, unordered_map[bint, vector[SearchResultPosting]]]]]()

        self.tier_disk_arrays = {}
        for path in os.listdir(os.path.join(self.index_path, self.name)):
            if path.startswith(TIER_PREFIX):
                try:
                    tier_id = int(path[len(TIER_PREFIX):])
                except ValueError:
                    continue
                self.tier_disk_arrays[tier_id] = DiskArray(os.path.join(self.index_path, self.name, path))
        self.num_tiers = len(self.tier_disk_arrays)
        
        self.global_document_frequencies = unordered_map[uint64_t, uint64_t]()
        cdef DiskArrayIterator it = self.tier_disk_arrays[0].iterator()
        cdef EntryInfo entry
        cdef uint64_t token_id = 0
        while it.has_next():
            entry = it.next_entry()
            self.global_document_frequencies[token_id] = entry.payload
            token_id += 1
        self.num_total_docs = NUM_TOTAL_DOCS
        self.average_field_lengths = vector[float]()
        for avg_len in AVG_FIELD_LENGTHS:
            self.average_field_lengths.push_back(avg_len)
        self.bm25_k = BM25_K
        self.bm25_bs = vector[float]()
        for b in BM25_BS:
            self.bm25_bs.push_back(b)
        self.bm25_field_boosts = vector[float]()
        for boost in BM25_FIELD_BOOSTS:
            self.bm25_field_boosts.push_back(boost)
        
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

        cdef dict config
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            with open(os.path.join(self.model_path, f"{self.name}.json"), "r") as f:
                config = json.load(f)
            self.ltr_model = ListNet(**config)
            self.ltr_model.load_state_dict(torch.load(os.path.join(self.model_path, f"{self.name}.pth"), map_location=self.device))
            self.ltr_model.eval()
        except Exception as e:
            print(f"Could not load LTR model for index {self.name}: {e}")
            self.ltr_model = None

        try:
            self.corpus_embeddings = torch.tensor(np.fromfile(os.path.join(self.embeddings_path, f"{self.name}.npy"), dtype="float32").reshape(-1, MAT_DIM))
            self.embed_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.embed_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
            self.embed_model.eval()
            self.embed_model.to(self.device)
        except Exception as e:
            print(f"Could not load embedding model for index {self.name}: {e}")
            self.corpus_embeddings = None
            self.embed_tokenizer = None
            self.embed_model = None

    cdef _mean_pooling(self, object model_output, object attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    cdef _embed_query(self, str query):
        sentences = [f"search query: {query}"]
        encoded_input = self.embed_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        encoded_input.to(self.device)
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :MAT_DIM]
        embeddings = F.normalize(embeddings, p=2, dim=1).squeeze(0)
        return embeddings

    cpdef object simulate_feature_matrix(self, list doc_ids, str query_text):
        cdef vector[uint64_t] query_tokens = vector[uint64_t]()
        cdef bytes query_bytes = query_text.encode("utf-8")
        cdef char* c_query = query_bytes
        cdef TokenizedField tokenized_field = self.tokenizer.tokenize(c_query, len(query_text), True)
        cdef vector[uint64_t] doc_ids_vector = vector[uint64_t]()
        cdef uint32_t i
        for i in range(len(doc_ids)):
            doc_ids_vector.push_back(<uint64_t>doc_ids[i])
        
        cdef object doc_embeddings = self.corpus_embeddings[doc_ids]
        cdef object query_embedding = self._embed_query(query_text)
        cdef object similarity_scores = torch.matmul(doc_embeddings, query_embedding)
        cdef vector[float] similarity_vector = vector[float]()
        for i in range(similarity_scores.size(0)):
            similarity_vector.push_back(<float>similarity_scores[i].item())

        cdef vector[SearchResultPosting] c_postings = self.simulate_search_result(doc_ids_vector, tokenized_field.tokens, similarity_vector)
        cdef float[:, :] features = get_features(
            tokenized_field.tokens, 
            c_postings, 
            len(doc_ids), 
            self.global_document_frequencies, 
            self.num_total_docs, 
            self.average_field_lengths, 
            self.bm25_k, 
            self.bm25_bs
        )
        return np.matrix(features)
    
    cdef vector[float] _on_the_fly_bm25(self, vector[uint64_t]& query_tokens, vector[SearchResultPosting]& postings, size_t top_k):
        cdef float[:, :] features = get_features(
            query_tokens, 
            postings, 
            top_k, 
            self.global_document_frequencies, 
            self.num_total_docs, 
            self.average_field_lengths, 
            self.bm25_k, 
            self.bm25_bs
        )
        cdef vector[float] bm25_scores = vector[float]()
        cdef uint32_t i
        for i in range(postings.size()):
            bm25_scores.push_back(features[i, 0] * self.bm25_field_boosts[0] + features[i, 1] * self.bm25_field_boosts[1])
        return bm25_scores

    cdef vector[SearchResultPosting] simulate_search_result(self, vector[uint64_t] doc_ids, vector[uint64_t]& query_tokens, vector[float]& similarity_scores):
        cdef vector[SearchResultPosting] result_postings = vector[SearchResultPosting]()
        cdef TokenizedDocument tokenized_doc
        cdef SearchResultPosting sr_posting
        cdef Posting posting
        cdef uint32_t i, j, k
        cdef uint64_t token

        cdef unordered_set[uint64_t] query_token_set = unordered_set[uint64_t]()
        for i in range(query_tokens.size()):
            query_token_set.insert(query_tokens[i])
        
        for i in range(doc_ids.size()):
            tokenized_doc = self.corpus.get_tokenized_document(<uint64_t>doc_ids[i], self.tokenizer)
            sr_posting.doc_id = <uint32_t>doc_ids[i]

            sr_posting.tokens = vector[uint64_t]()
            sr_posting.scores = vector[float]()
            sr_posting.total_score = 0.0
            sr_posting.similarity_score = similarity_scores[i]
            sr_posting.snippet_position = UINT32_MAX
            sr_posting.field_frequencies = vector[vector[uint32_t]]()
            sr_posting.field_lengths = vector[uint32_t]()
            sr_posting.char_positions = vector[vector[uint32_t]]()
        
            sr_posting.field_lengths = vector[uint32_t](tokenized_doc.num_fields)
            for j in range(tokenized_doc.num_fields):
                sr_posting.field_lengths[j] = tokenized_doc.field_lengths[j]
            
            for i in range(tokenized_doc.tokens.size()):
                token = tokenized_doc.tokens[i]
                posting = tokenized_doc.postings[i]
                
                if query_token_set.find(token) != query_token_set.end():
                    sr_posting.tokens.push_back(token)
                    sr_posting.scores.push_back(0.0)  # dummy score
                    sr_posting.num_fields = tokenized_doc.num_fields

                    sr_posting.field_frequencies.push_back(vector[uint32_t](tokenized_doc.num_fields))
                    k = sr_posting.field_frequencies.size() - 1
                    for j in range(tokenized_doc.num_fields):
                        sr_posting.field_frequencies[k][j] = posting.field_frequencies[j]
                    sr_posting.char_positions.push_back(posting.char_positions)
                free_posting(&posting, False)

            result_postings.push_back(sr_posting)
            free_tokenized_document(&tokenized_doc)
        
        cdef vector[float] bm25_scores = self._on_the_fly_bm25(query_tokens, result_postings, doc_ids.size())
        for i in range(result_postings.size()):
            result_postings[i].total_score = bm25_scores[i]

        return result_postings
    
    cdef uint32_t _tier_min_posting_list_length(self, QueryNode* node, uint32_t tier):
        
        cdef uint32_t min_length
        cdef uint64_t token
        cdef DiskArray tier_disk_array
        cdef EntryInfo entry
        cdef uint32_t posting_list_length

        with nogil:
            if node == NULL:
                return UINT32_MAX
            if node.left == NULL and node.right == NULL:
                min_length = UINT32_MAX
                with gil:
                    tier_disk_array = self.tier_disk_arrays[tier]
                for i in range(node.values.size()):
                    token = node.values[i]
                    entry = tier_disk_array.get(token)
                    posting_list_length = get_posting_list_length(entry.data, entry.length)
                    if posting_list_length < min_length:
                        min_length = posting_list_length
                return min_length
            with gil:
                return min(self._tier_min_posting_list_length(node.left, tier), self._tier_min_posting_list_length(node.right, tier))

    cdef uint32_t _min_tier_index(self, QueryNode* node, uint32_t top_k) noexcept nogil:
        cdef uint32_t tier
        cdef uint32_t min_length
        for tier in range(self.num_tiers):
            with gil:
                min_length = self._tier_min_posting_list_length(node, tier)
            if min_length >= top_k:
                return tier
        return self.num_tiers - 1

    cdef void _cache_postings(self, uint64_t token_id, uint32_t min_tier, uint32_t max_tier, bint with_positions, vector[SearchResultPosting]& postings) noexcept nogil:
        self.postings_cache[token_id][min_tier][max_tier][with_positions] = postings

    cdef bint _check_cache(self, uint64_t token_id, uint32_t min_tier, bint with_positions, uint32_t max_tier) noexcept nogil:
        return (self.postings_cache.find(token_id) != self.postings_cache.end() and self.postings_cache[token_id].find(min_tier) != self.postings_cache[token_id].end() and self.postings_cache[token_id][min_tier].find(max_tier) != self.postings_cache[token_id][min_tier].end() and self.postings_cache[token_id][min_tier][max_tier].find(with_positions) != self.postings_cache[token_id][min_tier][max_tier].end())
                
    cdef void _get_cached_postings(self, vector[SearchResultPosting]& postings, uint64_t token_id, uint32_t min_tier, bint with_positions, uint32_t max_tier) noexcept nogil:
        if self._check_cache(token_id, min_tier, with_positions, max_tier):
            postings = self.postings_cache[token_id][min_tier][max_tier][with_positions]
        else:
            postings = vector[SearchResultPosting]()
            
    cdef vector[SearchResultPosting] _get_postings(self, uint64_t token_id, uint32_t min_tier, uint32_t max_tier, bint with_positions):
        cdef vector[SearchResultPosting] sr_postings
        cdef vector[SearchResultPosting] tmp_sr_postings
        cdef EntryInfo entry
        cdef uint32_t current_tier, tier            
        cdef uint32_t start_reading_at = min_tier
        
        with nogil:

            if self._check_cache(token_id, min_tier, with_positions, max_tier):
                self._get_cached_postings(sr_postings, token_id, min_tier, with_positions, max_tier)
                with gil:
                    print(f"- Retrieved postings for token '{self.tokenizer.py_get(token_id)}': cache hit with {sr_postings.size()} postings from tiers {min_tier} to {max_tier}")
                return sr_postings

            for current_tier in range(max_tier, min_tier - 1 if min_tier > 0 else -1, -1):
                if self._check_cache(token_id, min_tier, with_positions, current_tier):
                    self._get_cached_postings(sr_postings, token_id, min_tier, with_positions, current_tier)
                    start_reading_at = current_tier + 1
                    with gil:
                        print(f"- Retrieved postings for token '{self.tokenizer.py_get(token_id)}': cache hit with {sr_postings.size()} postings from tiers {min_tier} to {current_tier}")
                    break

            for current_tier in range(start_reading_at, max_tier + 1):
                with gil:
                    disk_arr = <DiskArray>self.tier_disk_arrays[current_tier]
                entry = disk_arr.get(token_id)
                if entry.length > 0:
                    tmp_sr_postings = deserialize_search_result_postings(entry.data, entry.length, token_id, with_positions=with_positions)
                    with gil:
                        print(f"- Retrieving postings for token '{self.tokenizer.py_get(token_id)}': reading {tmp_sr_postings.size()} postings from tier {current_tier}")
                    union(sr_postings, tmp_sr_postings)
                else:
                    with gil:
                        print(f"- Retrieving postings for token '{self.tokenizer.py_get(token_id)}': no postings found in tier {current_tier}")
        if sr_postings.size() > 0:
            self._cache_postings(token_id, min_tier, max_tier, with_positions, sr_postings)
        
        return sr_postings
    
    cdef pair[CharPtr, uint32_t] _get_snippet(self, Document doc, SearchResultPosting posting):

        cdef int snippet_radius = SNIPPET_RADIUS
        cdef int position = <int>posting.snippet_position - doc.title_length if posting.snippet_position != UINT32_MAX else 1
        if position <= 0:
            position = 1
        cdef int start_pos = position - snippet_radius if position >= snippet_radius else 1
        cdef int end_pos = start_pos + 2 * snippet_radius if start_pos + 2 * snippet_radius < <int>doc.body_length else doc.body_length

        cdef char* snippet = <char*>malloc((end_pos - start_pos + 1) * sizeof(char))
        cdef int i
        cdef int first_space = 0, last_space = 0

        if start_pos != 0:
            for i in range(start_pos, end_pos):
                if doc.body[i] == ' ':
                    first_space = i
                    break
        if end_pos != <int>doc.body_length:
            for i in range(end_pos, start_pos, -1):
                if doc.body[i] == ' ':
                    last_space = i
                    break

        if first_space > 0:
            start_pos = first_space + 1
        if last_space > 0 and last_space > start_pos:
            end_pos = last_space

        for i in range(start_pos, end_pos):
            snippet[i - start_pos] = doc.body[i]
        snippet[end_pos - start_pos] = '\0'
        return pair[CharPtr, uint32_t](snippet, end_pos - start_pos)

    cdef vector[Document] _retrieve_documents_from_postings_with_snippets(self, vector[SearchResultPosting]& postings, vector[uint32_t] indices, uint32_t top_k) noexcept nogil:

        cdef vector[Document] documents = vector[Document]()
        cdef Document doc
        cdef float score = 0
        cdef pair[CharPtr, uint32_t] snippet_pair
        cdef size_t i
        if indices.size() <= 0:
            indices = vector[uint32_t]()
            for i in range(postings.size()):
                indices.push_back(i)

        for i in range(indices.size()):
            doc = self.corpus.get_document(postings[indices[i]].doc_id, lowercase=False)
            doc.score = postings[indices[i]].total_score
            with gil:
                snippet_pair = self._get_snippet(doc, postings[indices[i]])
            doc.snippet = snippet_pair.first
            doc.snippet_length = snippet_pair.second
            documents.push_back(doc)
            if documents.size() >= top_k:
                break
        return documents

    cdef pair[vector[SearchResultPosting], bint] _tiered_full_boolean_search(self, QueryNode* node, uint32_t min_k, uint32_t current_tier) noexcept nogil:

        if node == NULL:
            return pair[vector[SearchResultPosting], bint](vector[SearchResultPosting](), False)

        cdef vector[SearchResultPosting] result, other
        cdef pair[vector[SearchResultPosting], bint] left_res, right_res
        cdef bint result_isnot = False
        cdef uint64_t token
        cdef uint32_t local_tier = current_tier

        # leaf node; get at least min_k postings of that term
        if node.left == NULL and node.right == NULL:
            while True:
                if node.values.size() > 1:
                    with gil:
                        result = self._get_postings(node.values[0], 0, local_tier, True)
                    for i in range(1, node.values.size()):
                        token = node.values[i]
                        with gil:
                            other = self._get_postings(token, 0, local_tier, True)
                        intersection_phrase(result, other, 10)
                else:
                    with gil:
                        result = self._get_postings(node.values[0], 0, local_tier, False)
                local_tier += 1
                if result.size() >= min_k or local_tier >= self.num_tiers - 1:
                    break

            return pair[vector[SearchResultPosting], bint](result, False)
        
        if node.values[0] == self.not_operator:
            right_res = self._tiered_full_boolean_search(node.right, min_k, self.num_tiers - 1)
            right_res.second = not right_res.second
            return right_res

        left_res = self._tiered_full_boolean_search(node.left, min_k, local_tier)
        right_res = self._tiered_full_boolean_search(node.right, min_k, local_tier)

        if node.values[0] == self.and_operator:
            if not left_res.second and not right_res.second:
                intersection(left_res.first, right_res.first)
                result.swap(left_res.first)
                result_isnot = False
            elif left_res.second and not right_res.second:
                difference(right_res.first, left_res.first)
                result.swap(right_res.first)
                result_isnot = False
            elif not left_res.second and right_res.second:
                difference(left_res.first, right_res.first)
                result.swap(left_res.first)
                result_isnot = True
            else:
                union(left_res.first, right_res.first)
                result.swap(left_res.first)
                result_isnot = True
        elif node.values[0] == self.or_operator:
            if not left_res.second and not right_res.second:
                union(left_res.first, right_res.first)
                result.swap(left_res.first)
                result_isnot = False
            elif left_res.second and not right_res.second:
                difference(left_res.first, right_res.first)
                result.swap(left_res.first)
                result_isnot = True
            elif not left_res.second and right_res.second:
                difference(right_res.first, left_res.first)
                result.swap(right_res.first)
                result_isnot = True
            else:
                intersection(left_res.first, right_res.first)
                result.swap(left_res.first)
                result_isnot = True
        
        return pair[vector[SearchResultPosting], bint](result, result_isnot)
        
    cdef vector[uint32_t] _rank_documents(self, vector[uint64_t] query_tokens, vector[SearchResultPosting]& postings, int pre_select_k):
        cdef float[:, :] feature_matrix = get_features(
            query_tokens,
            postings, 
            pre_select_k,
            self.global_document_frequencies, 
            self.num_total_docs, 
            self.average_field_lengths, 
            self.bm25_k, 
            self.bm25_bs
        )
        cdef object input = torch.tensor([feature_matrix])
        cdef object output = self.ltr_model(input)[0]
        cdef list ranked_document_indices = list(torch.argsort(output, descending=True))
        cdef vector[uint32_t] ranked_documents = vector[uint32_t]()
        for i in range(len(ranked_document_indices)):
            ranked_documents.push_back(ranked_document_indices[i])
        return ranked_documents

    cdef TokenizedField _tokenize_query(self, str query):
        cdef bytes query_bytes = query.lower().encode("utf-8")
        cdef const char* query_c = query_bytes
        cdef TokenizedField tokenized_query = self.tokenizer.tokenize(query_c, len(query_bytes), True)
        print("- Tokens:", [self.tokenizer.py_get(tokenized_query.tokens[i]) for i in range(tokenized_query.tokens.size())])
        return tokenized_query

    cdef void _print_query_correction(self, vector[uint64_t] query_tokens):
        cdef pair[CharPtr, uint32_t] query_correction = self.spelling_corrector.get_top_correction(query_tokens, min_similarity=0.75)
        if query_correction.second > 0:
            print(f"Did you mean \033[4m{str(query_correction.first, 'utf-8')}\033[0m?")

    cdef vector[SearchResultPosting] _exact_search_postings(self, QueryNode* query_tree, size_t min_k, bint verbose) noexcept nogil:

        cdef uint32_t tier = self._min_tier_index(query_tree, min_k)
        if verbose:
            with gil:            
                print(f"- min tier: {tier}")
        cdef pair[vector[SearchResultPosting], bint] results_pair = pair[vector[SearchResultPosting], bint](vector[SearchResultPosting](), False)
        cdef clock_t start = clock()
        while results_pair.first.size() < min_k and tier < self.num_tiers:
            results_pair = self._tiered_full_boolean_search(query_tree, min_k, tier)
            if results_pair.first.size() < min_k and tier < self.num_tiers - 2:
                with gil:
                    print(f"- Not enough results {results_pair.first.size()}/{min_k} in tiers 0-{tier}, continuing search with tier {tier + 1}")
            tier += 1
        cdef clock_t end = clock()
        with gil:
            print(f"- retrieval inner took {(end - start) / CLOCKS_PER_SEC * 1000:.4f} milliseconds.")

        if verbose:
            with gil:            
                print(f"- max tier: {tier - 1}")
                print(f"- Number of results: {results_pair.first.size()}")
        return results_pair.first
    
    cdef vector[SearchResultPosting] _semantic_search_postings(self, str query, TokenizedField tokenized_query, size_t top_k):
        cdef object query_embedding = self._embed_query(query)
        cdef object similarity_scores = torch.matmul(self.corpus_embeddings, query_embedding)
        cdef object top_k_results = torch.topk(similarity_scores, k=top_k)
        cdef vector[uint64_t] top_k_indices = vector[uint64_t]()
        cdef vector[float] similarity_vector = vector[float]()
        cdef uint32_t i
        for i in range(top_k):
            top_k_indices.push_back(<uint64_t>top_k_results.indices[i].item())
            similarity_vector.push_back(<float>similarity_scores[top_k_results.indices[i]].item())
        return self.simulate_search_result(top_k_indices, tokenized_query.tokens, similarity_vector)

    cpdef object semantic_search(self, str query, size_t pre_select_k, size_t top_k, bint ltr_enabled):
        cdef TokenizedField tokenized_query = self._tokenize_query(query)
        cdef vector[SearchResultPosting] sr_postings = self._semantic_search_postings(query, tokenized_query, pre_select_k)
        if sr_postings.size() <= 0:
            return [], [], []

        cdef vector[uint32_t] ranking_indices = vector[uint32_t]()
        if ltr_enabled:
            ranking_indices = self._rank_documents(tokenized_query.tokens, sr_postings, top_k)
        
        cdef vector[Document] documents = self._retrieve_documents_from_postings_with_snippets(sr_postings, ranking_indices, top_k)
        cdef list result = [doc_to_dict(documents[i]) for i in range(min(top_k, sr_postings.size()))]
        cdef list semantic_scores = [sr_postings[i if not ltr_enabled else ranking_indices[i]].similarity_score for i in range(min(top_k, sr_postings.size()))]
        
        return result, ["semantic" for i in range(min(top_k, sr_postings.size()))], semantic_scores
        
        
    cpdef object exact_search(self, str query, size_t pre_select_k, size_t top_k, bint ltr_enabled):

        cdef TokenizedField tokenized_query = self._tokenize_query(query)
        cdef QueryNode* query_tree = self.query_parser.parse(tokenized_query.tokens)
        print_query_tree(query_tree, self.tokenizer, 0)
        self._print_query_correction(tokenized_query.tokens)

        cdef clock_t start = clock()
        cdef vector[SearchResultPosting] results_postings = self._exact_search_postings(query_tree, pre_select_k, verbose=True)
        if results_postings.size() <= 0:
            return [], [], []
        cdef clock_t end = clock()
        print(f"- retrieval took {(end - start) / CLOCKS_PER_SEC * 1000:.4f} milliseconds.")

        start = clock()
        sort(results_postings.begin(), results_postings.end(), compare_postings_scores)
        cdef vector[uint32_t] ranking_indices = vector[uint32_t]()
        if ltr_enabled:
            ranking_indices = self._rank_documents(tokenized_query.tokens, results_postings, pre_select_k)
        cdef vector[Document] documents = self._retrieve_documents_from_postings_with_snippets(results_postings, ranking_indices, top_k)
        end = clock()
        print(f"- ranking took {(end - start) / CLOCKS_PER_SEC * 1000:.4f} milliseconds.")

        cdef list results = [doc_to_dict(documents[i]) for i in range(min(top_k, results_postings.size()))]
        cdef list sources = ["exact" for i in range(min(top_k, results_postings.size()))]
        cdef list semantic_scores = [0 for i in range(min(top_k, results_postings.size()))]
        return results, sources, semantic_scores

    cpdef object combined_search(self, str query, size_t exact_search_preselect_k, size_t semantic_search_preselect_k, size_t top_k):

        cdef TokenizedField tokenized_query = self._tokenize_query(query)
        cdef QueryNode* query_tree = self.query_parser.parse(tokenized_query.tokens)
        self._print_query_correction(tokenized_query.tokens)

        cdef vector[SearchResultPosting] exact_postings = self._exact_search_postings(query_tree, exact_search_preselect_k, verbose=False)
        cdef vector[SearchResultPosting] semantic_postings = self._semantic_search_postings(query, tokenized_query, semantic_search_preselect_k)

        sort(exact_postings.begin(), exact_postings.end(), compare_postings_scores)

        cdef vector[SearchResultPosting] postings = vector[SearchResultPosting]()
        cdef size_t i
        cdef unordered_set[uint64_t] exact_doc_ids = unordered_set[uint64_t]()
        for i in range(min(exact_postings.size(), exact_search_preselect_k)):
            postings.push_back(exact_postings[i])
            exact_doc_ids.insert(<uint64_t>exact_postings[i].doc_id)
        
        sort(postings.begin(), postings.end(), compare_postings_doc_ids)
        sort(semantic_postings.begin(), semantic_postings.end(), compare_postings_doc_ids)

        cdef unordered_set[uint64_t] semantic_doc_ids = unordered_set[uint64_t]()
        for i in range(semantic_postings.size()):
            semantic_doc_ids.insert(<uint64_t>semantic_postings[i].doc_id)
        
        union(postings, semantic_postings)

        if postings.size() <= 0:
            return [], [], []

        cdef vector[uint32_t] ranking_indices = self._rank_documents(tokenized_query.tokens, postings, postings.size())
        cdef vector[Document] documents = self._retrieve_documents_from_postings_with_snippets(postings, ranking_indices, top_k)

        cdef list sources = []
        cdef list result = []
        cdef list semantic_scores = []
        for i in range(min(top_k, postings.size())):
            if exact_doc_ids.find(<uint64_t>documents[i].id) != exact_doc_ids.end() and semantic_doc_ids.find(<uint64_t>documents[i].id) != semantic_doc_ids.end():
                sources.append("exact & semantic")
            elif exact_doc_ids.find(<uint64_t>documents[i].id) != exact_doc_ids.end():
                sources.append("exact")
            else:
                sources.append("semantic")
            result.append(doc_to_dict(documents[i]))
            semantic_scores.append(postings[ranking_indices[i]].similarity_score)
        
        return result, sources, semantic_scores

        
    