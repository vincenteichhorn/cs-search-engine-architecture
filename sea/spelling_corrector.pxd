from sea.tokenizer cimport Tokenizer
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from libcpp.string cimport string as cstring

from libcpp.utility cimport pair

ctypedef char* CharPtr

cdef class SpellingCorrector:

    cdef Tokenizer tokenizer
    cdef unordered_map[uint64_t, uint32_t] token_freq_map
    cdef size_t exclude_threshold
    cdef unordered_map[uint64_t, vector[uint64_t]] kgram_index
    cdef size_t k

    cdef unordered_map[uint64_t, vector[uint64_t]] _build_kgram_index(self) noexcept nogil
    cdef vector[uint64_t] _get_bigram_hashes(self, cstring token) noexcept nogil
    cdef vector[uint64_t] get_candidates_tokens(self, cstring token) noexcept nogil
    cdef float _jaccard_similarity(self, vector[uint64_t] a, vector[uint64_t] b) noexcept nogil
    cdef pair[CharPtr, uint32_t] get_top_correction(self, vector[uint64_t] tokens, float min_similarity) noexcept nogil