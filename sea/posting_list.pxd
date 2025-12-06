from libcpp.vector cimport vector
from sea.document cimport Posting, free_posting
from libc.stdint cimport uint32_t
from libcpp cimport bool as cbool


cdef void intersection(vector[Posting]& self_items, vector[Posting]& other_items, cbool phrase) noexcept nogil
cdef void union(vector[Posting]& self_items, vector[Posting]& other_items) noexcept nogil
cdef void difference(vector[Posting]& self_items, vector[Posting]& other_items) noexcept nogil