from sea.util.gamma import pack_gammas, unpack_gammas
import time

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
    def deserialize(cls, data, only_doc_id=False):
        cdef int start = time.time()
        cdef list numbers
        numbers, remainder = unpack_gammas(data, read_n=1 if only_doc_id else -1)
        cdef int doc_id = numbers[0]
        cdef list position_list = [p-1 for p in numbers[1:]] if not only_doc_id else []
        cdef int end = time.time()
        print(f"Deserialized posting for doc_id {doc_id} in {(end - start)*1000:.8f} milliseconds")
        return cls(doc_id, position_list), remainder