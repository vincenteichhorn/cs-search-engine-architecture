from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.string cimport string as cstring
from sea.disk_array cimport DiskArray
from libcpp.unordered_map cimport unordered_map

cdef struct TokenizedField:
    uint32_t length
    vector[uint32_t] tokens
    vector[uint32_t] char_positions

cdef class Tokenizer:

    cdef str save_path
    cdef cset[cstring] stopwords
    cdef cset[cstring] query_stopwords
    cdef DiskArray disk_array
    cdef unordered_map[cstring, uint32_t] vocabulary
    cdef uint32_t max_token_id

    cdef object stemmer
    
    cdef inline bint _is_alphanum(self, char c)
    cdef inline char _to_lower(self, char c)
    cdef inline cstring _ascii_lower(self, cstring s)
    cdef inline bint _is_stopword(self, cstring word, bint is_query)
    cdef void _scan(self, cstring text, vector[cstring]& temp_tokens, vector[uint32_t]& temp_positions, bint is_query)
    cdef TokenizedField tokenize(self, bytes text, bint is_query)
    cpdef tuple py_tokenize(self, bytes text, bint is_query)
