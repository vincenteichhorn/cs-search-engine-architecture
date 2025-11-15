from sea.util.gamma import pack_gammas, unpack_gammas
import time
import struct
from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free

cpdef object posting_deserialize(
    const uint8_t[:] data,
    bint only_doc_id=False,
    object cls=None
):
    cdef unsigned int doc_id = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
    cdef unsigned int positions_len = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
    cdef Py_ssize_t bytes_read = 8 + positions_len * 4

    if cls is None:
        cls = Posting

    if only_doc_id:
        return cls(doc_id, []), bytes_read

    # Allocate a C array for positions
    cdef unsigned int[:] positions = <unsigned int[:positions_len]> malloc(positions_len * sizeof(unsigned int))
    if positions is None:
        raise MemoryError()

    cdef unsigned int i, cur = 8

    # Read positions directly into the C array
    for i in range(positions_len):
        positions[i] = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
        cur += 4

    # Return Posting object; convert to list only if necessary
    return cls(doc_id, positions), bytes_read

cdef class Posting:
    cdef public int doc_id
    cdef public object positions

    def __cinit__(self, int doc_id, object positions):
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
        return posting_deserialize(data, only_doc_id, cls)