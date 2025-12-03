from libc.stdint cimport uint64_t, uint32_t
from libcpp.string cimport string as cstring
from libcpp.vector cimport vector

cdef struct Document:
    uint64_t id
    char* url
    uint32_t url_length
    char* title
    uint32_t title_length
    char* body
    uint32_t body_length

cdef struct Posting:
    uint32_t doc_id_diff
    uint32_t* field_frequencies
    vector[uint32_t] char_positions
    vector[uint32_t] token_positions

cdef struct TokenizedDocument:
    uint64_t id
    uint32_t num_fields
    uint32_t* field_lengths
    vector[uint32_t] tokens
    vector[Posting] postings
