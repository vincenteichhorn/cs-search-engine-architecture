from libc.stdint cimport uint32_t, uint64_t, uint8_t
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string as cstring
from sea.util.disk_array cimport DiskArray
from libcpp.unordered_map cimport unordered_map
from sea.util.fast_stemmer cimport FastStemmer

cdef uint64_t fnv1a_hash(const char* ptr, size_t length) noexcept nogil

cdef struct TokenizedField:
    uint32_t length
    uint32_t max_token_id
    vector[uint64_t] tokens
    vector[uint32_t] char_positions

cdef class Tokenizer:

    cdef str save_path
    cdef unordered_set[uint64_t] stopwords
    cdef unordered_set[uint64_t] query_stopwords
    cdef DiskArray disk_array
    cdef public unordered_map[uint64_t, uint64_t] vocabulary
    cdef uint64_t max_token_id

    cdef bint stem
    cdef FastStemmer stemmer

    cdef unordered_map[uint64_t, uint64_t] _load_vocabulary(self)
    
    cdef inline bint _is_stopword(self, uint64_t token_hash, bint is_query) noexcept nogil
    cdef void _scan(self, const char* text, uint32_t length, vector[const char*]& token_ptrs, vector[uint32_t]& token_lens, vector[uint32_t]& char_positions, bint is_query) noexcept nogil
    
    cdef TokenizedField tokenize(self, const char* text, uint32_t length, bint is_query) noexcept nogil
    cpdef tuple py_tokenize(self, bytes text, bint is_query)
    
    cdef const cstring get(self, uint64_t idx) noexcept nogil
    cpdef str py_get(self, uint64_t idx)

    cpdef uint64_t py_vocab_lookup(self, bytes token)
    cdef uint64_t vocab_lookup(self, const char* token, uint32_t length) noexcept nogil
    
    cpdef void flush(self)
    cdef void _flush(self) noexcept nogil
