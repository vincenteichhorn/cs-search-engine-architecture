from sea.util.gamma import pack_gammas, unpack_gammas
import time
import struct
from cpython.list cimport PyList_New, PyList_SET_ITEM
from cpython.long cimport PyLong_FromLong
from libc.stdint cimport uint8_t

cpdef object deserialize_fast_impl(
    const uint8_t[:] data,
    bint only_doc_id=False,
    object cls=None
):
    cdef unsigned int doc_id = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
    cdef unsigned int positions_len = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        
    cdef int bytes_read = 8 + positions_len * 4

    if cls is None:
        cls = Posting
    
    if only_doc_id:
        return cls(doc_id, []), bytes_read

    cdef object pylist = PyList_New(positions_len)
    cdef unsigned int i
    cdef int pos
    cdef Py_ssize_t cur = 8

    for i in range(positions_len):
        pos = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
        PyList_SET_ITEM(pylist, i, PyLong_FromLong(pos))
        cur += 4

    return cls(doc_id, <list>pylist), bytes_read

cdef class Posting:
    cdef public int doc_id
    cdef public list positions

    def __cinit__(self, int doc_id, list positions):
        self.doc_id = doc_id
        self.positions = positions

    cpdef void serialize_gamma(self, object writer):
        pack_gammas(writer, [self.doc_id] + [p+1 for p in self.positions])

    @classmethod
    def deserialize_gamma(cls, object reader, bint only_doc_id=False):
        cdef list numbers
        numbers = unpack_gammas(reader, read_n=1 if only_doc_id else -1)

        cdef int doc_id = numbers[0]
        cdef list positions = [p-1 for p in numbers[1:]] if not only_doc_id else []
        return cls(doc_id, positions)

    cpdef bytes serialize(self):
        cdef bytearray buffer = bytearray()
        buffer.extend(struct.pack(">I", self.doc_id))
        buffer.extend(struct.pack(">I", len(self.positions)))
        for p in self.positions:
            buffer.extend(struct.pack(">I", p))
        return bytes(buffer)

    @classmethod
    def deserialize(cls, const uint8_t[:] data, bint only_doc_id=False):
        return deserialize_fast_impl(data, only_doc_id, cls)