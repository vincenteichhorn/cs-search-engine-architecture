from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef uint64_t read_uint64(const uint8_t* buf, Py_ssize_t offset)
cdef uint32_t read_uint32(const uint8_t* buf, Py_ssize_t offset)

cdef class SmartBuffer:
    cdef uint8_t* ptr
    cdef size_t size
    cpdef uint8_t[:] to_view(self)
    cpdef bytearray to_bytearray(self)
