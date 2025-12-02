from sea.disk_array cimport DiskArray
from sea.document cimport Document, TokenizedDocument
from sea.util.memory cimport SmartBuffer
from sea.tokenizer cimport Tokenizer
from libc.stdint cimport uint8_t, uint64_t

cpdef str identity_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length)
cpdef Document document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length) noexcept nogil
cpdef TokenizedDocument tokenized_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, Tokenizer tokenizer) noexcept nogil

cdef class Corpus:
    cdef str save_path
    cdef str data_file_path
    cdef DiskArray disk_array

    cdef object data_file
    cdef uint64_t data_size
    cdef const uint8_t[:] data_map
    cdef uint64_t data_offset

    cpdef object get(self, uint64_t idx, object processor)
    cpdef void flush(self)
    cpdef object next(self, object processor)