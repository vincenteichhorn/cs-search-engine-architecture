from sea.corpus cimport Corpus, doc_to_dict
from sea.tokenizer cimport Tokenizer, TokenizedField
from sea.document cimport Document, Posting, TokenizedDocument, SearchResultPosting, deserialize_postings, deserialize_search_result_postings, free_posting, free_tokenized_document_with_postings, free_tokenized_document, get_posting_list_length, create_search_result_postings
from sea.posting_list cimport intersection, intersection_phrase, union, difference
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
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t
import numpy as np
from libcpp.unordered_set cimport unordered_set
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

cdef str TIER_PREFIX = "tier_"
cdef size_t SNIPPET_RADIUS = 150
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

    cdef dict tier_disk_arrays
    cdef uint32_t num_tiers
    cdef uint64_t and_operator
    cdef uint64_t or_operator
    cdef uint64_t not_operator

    cdef uint64_t num_total_docs
    cdef vector[float] average_field_lengths
    cdef float bm25_k
    cdef vector[float] bm25_bs

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
            sr_posting.snippet_position = 0
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

        return result_postings
    
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
            # print(f"Tier {tier} has min posting list length {min_length}")
            if min_length >= top_k:
                return tier
        return self.num_tiers - 1

    cdef vector[SearchResultPosting] _get_postings(self, uint64_t token_id, uint32_t min_tier, uint32_t max_tier):
        
        cdef DiskArray tier_disk_array
        cdef EntryInfo entry
        cdef vector[SearchResultPosting] sr_postings, tmp_sr_postings
        cdef vector[Posting] tmp_postings
        for tier in range(min_tier, max_tier + 1):
            tier_disk_array = self.tier_disk_arrays[tier]
            entry = tier_disk_array.get(token_id)
            # tmp_postings = deserialize_postings(entry.data, entry.length)
            tmp_sr_postings = deserialize_search_result_postings(entry.data, entry.length, token_id)
            union(sr_postings, tmp_sr_postings)
        return sr_postings
    
    cdef pair[CharPtr, uint32_t] _get_snippet(self, Document doc, SearchResultPosting posting):

        cdef size_t snippet_length = SNIPPET_RADIUS
        cdef uint32_t l
        cdef uint32_t position = doc.title_length + snippet_length // 2
        if posting.char_positions.size() != 0:
            l = posting.char_positions.size() - 1
            position = posting.char_positions[l][0] - doc.title_length if posting.char_positions[l].size() > 0 else 0
        if posting.snippet_position != 0:
            position = posting.snippet_position - doc.title_length
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
    

    cdef pair[vector[SearchResultPosting], bint] _full_boolean_search(self, QueryNode* node, uint32_t min_tier, uint32_t max_tier):


        if node == NULL:
            return pair[vector[SearchResultPosting], bint](vector[SearchResultPosting](), False)
        cdef vector[SearchResultPosting] left_postings, right_postings, postings
        cdef pair[vector[SearchResultPosting], bint] left_pair, right_pair, tmp_pair
        cdef uint64_t token, i
    
        if node.left == NULL and node.right == NULL:
            if node.values.size() > 1:
                result = self._get_postings(node.values[0], min_tier, max_tier)
                for i in range(1, node.values.size()):
                    token = node.values[i]
                    other_posting_list = self._get_postings(token, min_tier, max_tier)
                    intersection_phrase(result, other_posting_list, 10)
                return pair[vector[SearchResultPosting], bint](result, False)
            else:
                return pair[vector[SearchResultPosting], bint](self._get_postings(node.values[0], min_tier, max_tier), False)

        if node.values[0] == self.not_operator:
            tmp_pair = self._full_boolean_search(node.right, min_tier, max_tier)
            tmp_pair.second = not tmp_pair.second
            return tmp_pair

        left_pair = self._full_boolean_search(node.left, min_tier, max_tier)
        right_pair = self._full_boolean_search(node.right, min_tier, max_tier)

        if node.values[0] == self.and_operator:
            if not left_pair.second and not right_pair.second:
                intersection(left_pair.first, right_pair.first)
                return pair[vector[SearchResultPosting], bint](left_pair.first, False)
            elif left_pair.second and not right_pair.second:
                difference(right_pair.first, left_pair.first)
                return pair[vector[SearchResultPosting], bint](right_pair.first, False)
            elif not left_pair.second and right_pair.second:
                difference(left_pair.first, right_pair.first)
                return pair[vector[SearchResultPosting], bint](left_pair.first, False)
            else:
                union(left_pair.first, right_pair.first)
                return pair[vector[SearchResultPosting], bint](left_pair.first, True)
        elif node.values[0] == self.or_operator:
            if not left_pair.second and not right_pair.second:
                union(left_pair.first, right_pair.first)
                return pair[vector[SearchResultPosting], bint](left_pair.first, False)
            elif left_pair.second and not right_pair.second:
                difference(left_pair.first, right_pair.first)
                return pair[vector[SearchResultPosting], bint](left_pair.first, True)
            elif not left_pair.second and right_pair.second:
                difference(right_pair.first, left_pair.first)
                return pair[vector[SearchResultPosting], bint](right_pair.first, True)
            else:
                intersection(left_pair.first, right_pair.first)
                return pair[vector[SearchResultPosting], bint](left_pair.first, True)
        else:
            return pair[vector[SearchResultPosting], bint](vector[SearchResultPosting](), False)
    
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

    cpdef list semantic_search(self, str query, size_t top_k):

        cdef bytes query_bytes = query.lower().encode("utf-8")
        cdef const char* query_c = query_bytes
        cdef TokenizedField tokenized_query = self.tokenizer.tokenize(query_c, len(query_bytes), True)

        cdef object query_embedding = self._embed_query(query)
        cdef object similarity_scores = torch.matmul(self.corpus_embeddings, query_embedding)
        cdef vector[float] similarity_vector = vector[float]()
        cdef uint32_t i
        for i in range(similarity_scores.size(0)):
            similarity_vector.push_back(<float>similarity_scores[i].item())
        cdef object top_k_results = torch.topk(similarity_scores, k=top_k)
        cdef vector[uint64_t] top_k_indices = vector[uint64_t]()
        for i in range(top_k):
            top_k_indices.push_back(<uint64_t>top_k_results.indices[i].item())
        cdef vector[SearchResultPosting] sr_postings = self.simulate_search_result(top_k_indices, tokenized_query.tokens, similarity_vector)
        cdef vector[Document] documents = self._retrieve_documents_from_postings_with_snippets(sr_postings, vector[uint32_t](), top_k)
        for i in range(documents.size()):
            documents[i].score = top_k_results.values[<int>i].item()
        cdef list results = [doc_to_dict(documents[i]) for i in range(min(top_k, sr_postings.size()))]
        return results
        
        
    cpdef list search(self, str query, size_t pre_select_k, size_t top_k):

        cdef bytes query_bytes = query.lower().encode("utf-8")
        cdef const char* query_c = query_bytes
        cdef TokenizedField tokenized_query = self.tokenizer.tokenize(query_c, len(query_bytes), True)

        cdef pair[CharPtr, uint32_t] query_correction = self.spelling_corrector.get_top_correction(tokenized_query.tokens, min_similarity=0.75)
        if query_correction.second > 0:
            print(f"Did you mean \033[4m{str(query_correction.first, 'utf-8')}\033[0m?")

        cdef QueryNode* query_tree = self.query_parser.parse(tokenized_query.tokens)
        # print("- Query Tree:")
        # print_query_tree(query_tree, self.tokenizer, 0)

        cdef uint32_t tier = self._min_tier_index(query_tree, pre_select_k)

        cdef list results = []
        cdef pair[vector[SearchResultPosting], bint] results_pair = pair[vector[SearchResultPosting], bint](vector[SearchResultPosting](), False)
        while results_pair.first.size() < pre_select_k and tier < self.num_tiers:
            results_pair = self._full_boolean_search(query_tree, min_tier=0, max_tier=tier)
            tier += 1

        print(f"- Gone up to tier {tier-1}")
        print(f"- Number of results: {results_pair.first.size()}")
        if results_pair.first.size() <= 0:
            return results

        sort(results_pair.first.begin(), results_pair.first.end(), compare_postings_scores)
        cdef vector[uint32_t] ranking_indices = self._rank_documents(tokenized_query.tokens, results_pair.first, pre_select_k)
        cdef vector[Document] documents = self._retrieve_documents_from_postings_with_snippets(results_pair.first, ranking_indices, top_k)
        results = [doc_to_dict(documents[i]) for i in range(min(top_k, results_pair.first.size()))]

        return results