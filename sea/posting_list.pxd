from libcpp.vector cimport vector
from sea.document cimport SearchResultPosting
from libcpp cimport bool as cbool
from libc.stdint cimport uint32_t

cdef void intersection(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil
cdef void intersection_phrase(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items, uint32_t k) noexcept nogil
cdef void union(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil
cdef void difference(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil