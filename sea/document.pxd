from libc.stdint cimport uint64_t, uint32_t
from libcpp.string cimport string as cstring
from sea.tokenizer cimport TokenizedField
from libcpp.vector cimport vector

cdef struct Document:
    uint64_t id
    cstring title
    cstring body
    cstring url

cdef struct TokenInfo:
    uint32_t char_position
    uint32_t token_position
    uint32_t frequency

cdef struct TokenizedDocument:
    uint64_t id
    vector[uint32_t] field_lengths
    vector[uint32_t] tokens
    vector[TokenInfo] token_infos
