from sea.util.gamma import pack_gammas, unpack_gammas
import time
import struct

cdef class Posting:
    cdef public int doc_id
    cdef public list positions

    def __cinit__(self, int doc_id, list positions):
        self.doc_id = doc_id
        self.positions = positions

    # cpdef bytes serialize(self):
    #     cdef bytes data = pack_gammas([self.doc_id] + [p+1 for p in self.positions])
    #     return data

    # @classmethod
    # def deserialize(cls, object reader, bint only_doc_id=False):
    #     cdef list numbers
    #     numbers = unpack_gammas(reader, read_n=1 if only_doc_id else -1)

    #     cdef int doc_id = numbers[0]
    #     cdef list positions = [p-1 for p in numbers[1:]] if not only_doc_id else []
    #     return cls(doc_id, positions)

    cpdef bytes serialize(self):
        cdef bytearray buffer = bytearray()
        buffer.extend(struct.pack(">I", self.doc_id))
        buffer.extend(struct.pack(">I", len(self.positions)))
        for p in self.positions:
            buffer.extend(struct.pack(">I", p))
        return bytes(buffer)

    @classmethod
    def deserialize(cls, bytes data, bint only_doc_id=False):
        cdef list positions
        cdef int positions_len
        cdef int doc_id = struct.unpack(">I", data[0:4])[0]
        if only_doc_id:
            positions = []
        else:
            positions_len = struct.unpack(">I", data[4:8])[0]
            positions = [struct.unpack(">I", data[8+i*4:12+i*4])[0] for i in range(positions_len)]
        cdef bytes remainder = data[8+positions_len*4:]
        return cls(doc_id, positions), remainder