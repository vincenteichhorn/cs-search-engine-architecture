from libcpp.vector cimport vector
from sea.document cimport SearchResultPosting
from libcpp cimport bool as cbool


cdef void intersection(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items, cbool phrase) noexcept nogil
cdef void union(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil
cdef void difference(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil