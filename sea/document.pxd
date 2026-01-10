from libc.stdint cimport uint64_t, uint32_t, int64_t, UINT64_MAX, uint8_t
from libcpp.string cimport string as cstring
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool as cbool

ctypedef uint8_t* BytePtr
cdef cbool compare_postings_ptr(const Posting* a, const Posting* b) noexcept nogil
cdef void free_document(Document* doc) noexcept nogil
cdef void free_posting(Posting* posting, cbool all) noexcept nogil
cdef void free_tokenized_document(TokenizedDocument* tokenized_doc) noexcept nogil
cdef void free_tokenized_document_with_postings(TokenizedDocument* tokenized_doc) noexcept nogil
cdef pair[BytePtr, uint32_t] serialize_postings(vector[Posting*]* postings) noexcept nogil
cdef uint32_t get_posting_list_length(const uint8_t* data, uint32_t length) noexcept nogil
cdef vector[Posting] deserialize_postings(const uint8_t* data, uint32_t length) noexcept nogil
cdef pair[float, size_t] update_posting_score(const uint8_t* data, size_t offset, float idf, float bm25k, vector[float]& field_boosts, vector[float]& bm25_bs, vector[float]& avg_field_lengths) noexcept nogil

cdef struct Document:
    uint64_t id
    float score
    char* url
    uint32_t url_length
    char* title
    uint32_t title_length
    char* body
    uint32_t body_length
    char* snippet
    uint32_t snippet_length

cdef struct Posting:
    uint32_t doc_id
    float score
    uint32_t num_fields
    uint32_t* field_frequencies
    uint32_t* field_lengths
    vector[uint32_t] char_positions
    # vector[uint32_t] token_positions

cdef struct TokenizedDocument:
    uint64_t id
    uint32_t num_fields
    uint32_t* field_lengths
    vector[uint64_t] tokens
    vector[Posting] postings

