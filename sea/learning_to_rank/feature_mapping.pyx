# cython: boundscheck=False, wraparound=False, cdivision=True
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from sea.document cimport SearchResultPosting
import numpy as np
from libcpp.unordered_map cimport unordered_map
from libc.math cimport log  
from libc.float cimport FLT_MAX

cdef get_features(vector[uint64_t]& query_tokens, vector[SearchResultPosting]& postings, int top_k, unordered_map[uint64_t, uint64_t]& doc_freqs, uint64_t num_total_docs, vector[float]& average_field_lengths, float bm25_k, vector[float]& bm25_bs):

    cdef int num_features = 7
    cdef uint32_t num_docs = min(postings.size(), <size_t>top_k)
    cdef uint64_t num_query_tokens = query_tokens.size()
    cdef float[:, :] features = np.zeros((num_docs, num_features), dtype=np.float32)
    cdef uint64_t i, j, k, t

    # columns:
    # 0: BM25 title
    # 1: BM25 body
    # 2: title length
    # 3: body length
    # 4: ratio of query tokens in title
    # 5: ratio of query tokens in body
    # 6: first occurrence in document
    
    for i in range(num_docs):
        #large initial value for first occurrence
        features[i, 6] = FLT_MAX
        for j in range(postings[i].tokens.size()):
            current_token = postings[i].tokens[j]
            current_idf = log((num_total_docs - doc_freqs[current_token] + 0.5) / (doc_freqs[current_token] + 0.5))

            # title field
            current_tf = postings[i].field_frequencies[j][0] # count the term frequency in title
            features[i, 0] += current_idf * ((current_tf * (bm25_k + 1)) / (current_tf + bm25_k * (1 - bm25_bs[0] + bm25_bs[0] * (postings[i].field_lengths[0] / average_field_lengths[0]))))
            features[i, 4] += 1.0/num_query_tokens if num_query_tokens > 0 else 0.0 # ratio of query tokens in title
            
            # body field
            current_tf = postings[i].field_frequencies[j][1]
            features[i, 1] += current_idf * ((current_tf * (bm25_k + 1)) / (current_tf + bm25_k * (1 - bm25_bs[1] + bm25_bs[1] * (postings[i].field_lengths[1] / average_field_lengths[1]))))
            features[i, 5] += 1.0/num_query_tokens if num_query_tokens > 0 else 0.0 # ratio of query tokens in body
        
            features[i, 2] = postings[i].field_lengths[0] # title length
            features[i, 3] = postings[i].field_lengths[1] # body length
            features[i, 6] = min(features[i, 6], postings[i].char_positions[j][0]) # first occurrence in document
            
        if features[i, 6] == FLT_MAX:
            features[i, 6] = -1

    return features