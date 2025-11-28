from sea.util.gamma import pack_gammas, unpack_gammas
import time
import struct
from libc.stdint cimport uint8_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy



cpdef object posting_deserialize(
    const uint8_t[:] data,
    object cls=None,
    int offset=0
):
    cdef unsigned int i, cur = offset

    cdef unsigned int doc_id = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
    cur += 4
    cdef float score = struct.unpack(">f", data[cur:cur+4])[0]
    cur += 4
    cdef unsigned int num_fields = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
    cur += 4
    cdef unsigned int[:] field_freqs = <unsigned int[:num_fields]> malloc(num_fields * sizeof(unsigned int))
    cdef unsigned int[:] field_lengths = <unsigned int[:num_fields]> malloc(num_fields * sizeof(unsigned int))
    if field_freqs is None or field_lengths is None:
        raise MemoryError()


    for i in range(num_fields):
        field_freqs[i] = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
        cur += 4

    for i in range(num_fields):
        field_lengths[i] = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
        cur += 4

    cdef unsigned int positions_len = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
    cdef unsigned int[:] char_positions = <unsigned int[:positions_len]> malloc(positions_len * sizeof(unsigned int))
    cur += 4
    cdef Py_ssize_t bytes_read = 16 + num_fields * 4 + num_fields * 4 + positions_len * 4

    for i in range(positions_len):
        char_positions[i] = (data[cur] << 24) | (data[cur+1] << 16) | (data[cur+2] << 8) | data[cur+3]
        cur += 4

    if cls is None:
        cls = Posting

    # Return Posting object; convert to list only if necessary
    return cls(doc_id, char_positions, field_freqs, field_lengths, score), bytes_read

cdef class Posting:
    cdef public int doc_id
    cdef public object char_positions
    cdef public object field_freqs
    cdef public object field_lengths
    cdef public float score

    def __cinit__(self, int doc_id, object char_positions, object freqs, object field_lengths, float score = 0):
        self.doc_id = doc_id
        self.char_positions = char_positions
        self.field_freqs = freqs
        self.field_lengths = field_lengths
        self.score = score

    cpdef bytes serialize(self):
        cdef bytearray buffer = bytearray()
        buffer.extend(struct.pack(">I", self.doc_id))
        buffer.extend(struct.pack(">f", self.score))
        buffer.extend(struct.pack(">I", len(self.field_freqs)))
        for f in self.field_freqs:
            buffer.extend(struct.pack(">I", f))
        for l in self.field_lengths:
            buffer.extend(struct.pack(">I", l))
        buffer.extend(struct.pack(">I", len(self.char_positions)))
        for cp in self.char_positions:
            buffer.extend(struct.pack(">I", cp))
        return bytes(buffer)

    @classmethod
    def deserialize(cls, const uint8_t[:] data):
        return posting_deserialize(data, cls)