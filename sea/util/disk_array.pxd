from libcpp.vector cimport vector
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.stdio cimport FILE
from libcpp cimport bool as cbool

ctypedef struct EntryInfo:
    uint64_t payload
    const uint8_t* data
    uint32_t length

ctypedef const uint8_t* BytePtr

cdef class DiskArray:

    cdef vector[uint64_t] payloads
    cdef vector[uint64_t] data_offsets
    cdef vector[uint32_t] data_lengths
    cdef uint8_t* data_buffer
    cdef public size_t data_buffer_size
    cdef size_t data_buffer_capacity

    cdef str path
    cdef cbool open_read_maps

    cdef str index_file_path
    cdef bytes index_file_path_bytes
    cdef const char* index_file_path_c
    cdef int index_fd
    cdef FILE* index_file_ptr
    cdef size_t index_size
    cdef BytePtr index_read_buffer
    cdef BytePtr index_read_mmap

    cdef str data_file_path
    cdef bytes data_file_path_bytes
    cdef const char* data_file_path_c
    cdef int data_fd
    cdef FILE* data_file_ptr
    cdef size_t data_size
    cdef BytePtr data_read_buffer
    cdef size_t chunk_size
    cdef BytePtr data_read_mmap

    cdef uint32_t entry_size
    cdef uint64_t current_disk_offset
    cdef uint64_t current_idx
    cdef uint64_t current_disk_idx

    cdef void _open_read_maps(self) noexcept nogil

    cpdef uint64_t py_append(self, uint64_t payload, bytes data)
    cdef uint64_t append(self, uint64_t payload, const uint8_t* data, uint64_t offset, uint32_t length) noexcept nogil

    cpdef uint64_t py_add_to_last(self, bytes data)
    cdef uint64_t add_to_last(self, const uint8_t* data, uint64_t offset, uint32_t length) noexcept nogil

    cdef void append_large_flush(self, uint64_t payload, const uint8_t* data, uint64_t offset, uint32_t length) noexcept nogil

    cpdef uint64_t size(self)
    
    cpdef object py_get(self, uint64_t idx)
    cdef EntryInfo get(self, uint64_t idx) noexcept nogil
    
    cpdef void flush(self)
    cdef void _flush(self) noexcept nogil
    cpdef DiskArrayIterator iterator(self)


cdef class DiskArrayIterator:
    cdef DiskArray disk_array
    cdef uint64_t idx
    cdef uint64_t size
    cdef EntryInfo current_entry

    cdef cbool has_next(self) noexcept nogil
    cdef EntryInfo next_entry(self) noexcept nogil
    cdef EntryInfo current(self) noexcept nogil