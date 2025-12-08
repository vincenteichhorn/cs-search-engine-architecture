from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef uint64_t read_uint64(const uint8_t* buf, Py_ssize_t offset) noexcept nogil
cdef uint32_t read_uint32(const uint8_t* buf, Py_ssize_t offset) noexcept nogil
cdef size_t get_memory_usage() noexcept nogil