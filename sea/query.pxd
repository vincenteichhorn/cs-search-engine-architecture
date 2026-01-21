from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from sea.tokenizer cimport Tokenizer


cdef cppclass QueryNode:
    QueryNode* left
    QueryNode* right
    vector[uint64_t] values

ctypedef QueryNode* QueryNodePtr

cdef void print_query_tree(QueryNode* node, Tokenizer tokenizer, uint64_t depth) noexcept nogil
cdef dict query_tree_to_dict(QueryNode* node)

cdef class QueryParser:

    cdef uint64_t and_operator
    cdef uint64_t or_operator
    cdef uint64_t not_operator
    cdef uint64_t open_paren
    cdef uint64_t close_paren
    cdef uint64_t phrase_marker
    cdef unordered_map[uint64_t, uint32_t] operator_precedence

    cpdef dict py_parse(self, list tokens)
    cdef QueryNode* parse(self, vector[uint64_t]& tokens) noexcept nogil
    cdef void _remove_surrounding_operators(self, vector[uint64_t]& tokens) noexcept nogil
    cdef void _remove_consecutive_operators(self, vector[uint64_t]& tokens) noexcept nogil
    cdef void _fill_implicit_ands(self, vector[uint64_t]& tokens) noexcept nogil
    cdef void _remove_ands_in_phrases(self, vector[uint64_t]& tokens) noexcept nogil