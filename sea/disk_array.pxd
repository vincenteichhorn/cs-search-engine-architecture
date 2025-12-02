from libcpp.vector cimport vector
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from sea.util.memory cimport SmartBuffer

cdef class DiskArray:
    cdef vector[uint64_t] data_offsets
    cdef vector[uint32_t] data_lengths
    cdef vector[uint8_t] data_buffer

    cdef str path

    cdef str index_file_path
    cdef object index_file_read
    cdef object index_file_write
    cdef size_t index_size
    cdef const uint8_t[:] index_map

    cdef str data_file_path
    cdef object data_file_read
    cdef object data_file_write
    cdef size_t data_size
    cdef const uint8_t[:] data_map

    cdef uint32_t entry_size
    cdef uint64_t current_offset
    cdef uint64_t current_disk_offset
    cdef uint64_t current_idx
    cdef uint64_t current_disk_idx

    cdef void _open_maps(self)
    cpdef uint64_t py_append(self, bytes data)
    cdef uint64_t append(self, const uint8_t* data, uint64_t offset, uint64_t length) noexcept nogil
    cpdef uint64_t size(self)
    cpdef bytes py_get(self, uint64_t idx)
    cdef const uint8_t[:] get(self, uint64_t idx) noexcept nogil
    cpdef void flush(self)
