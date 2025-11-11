from sea.util.gamma import pack_gammas, unpack_gammas

cdef class Posting:
    cdef public int doc_id
    cdef public list positions

    def __cinit__(self, int doc_id, list positions):
        self.doc_id = doc_id
        self.positions = positions

    cpdef bytes serialize(self):
        cdef bytes data = pack_gammas([self.doc_id] + [p+1 for p in self.positions])
        return data

    @classmethod
    def deserialize(cls, data):
        cdef list numbers
        numbers, remainder = unpack_gammas(data)
        cdef int doc_id = numbers[0]
        cdef list position_list = [p-1 for p in numbers[1:]]
        return cls(doc_id, position_list), remainder