from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free

cdef uint64_t read_uint64(const uint8_t* buf, Py_ssize_t offset):
    cdef const uint64_t* p = <const uint64_t*>(&buf[offset])
    return p[0]

cdef uint32_t read_uint32(const uint8_t* buf, Py_ssize_t offset):
    cdef const uint32_t* p = <const uint32_t*>(&buf[offset])
    return p[0]

cdef class SmartBuffer:

    def __cinit__(self, const uint8_t* data, uint64_t size, uint64_t offset=0):
        self.size = size
        self.ptr = <uint8_t *>malloc(size)
        if self.ptr == NULL:
            raise MemoryError("Unable to allocate memory for SmartBuffer")
        for i in range(size):
            self.ptr[i] = data[offset + i]

    def __dealloc__(self):
        if self.ptr != NULL:
            free(self.ptr)
            self.ptr = NULL

    cpdef uint8_t[:] to_view(self):
        return <uint8_t[:self.size]>self.ptr # type: ignore

    cpdef bytearray to_bytearray(self):
        cdef bytearray result = bytearray(self.size)
        for i in range(self.size):
            result[i] = self.ptr[i]
        return result