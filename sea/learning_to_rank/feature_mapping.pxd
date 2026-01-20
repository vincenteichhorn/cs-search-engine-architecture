from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from sea.document cimport SearchResultPosting
from libcpp.unordered_map cimport unordered_map

cdef get_features(vector[uint64_t]& query_tokens, vector[SearchResultPosting]& postings, int top_k, unordered_map[uint64_t, uint64_t]& doc_freqs, uint64_t num_total_docs, vector[float]& average_field_lengths, float bm25_k, vector[float]& bm25_bs)