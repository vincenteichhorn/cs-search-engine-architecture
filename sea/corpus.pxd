from sea.util.disk_array cimport DiskArray
from sea.document cimport Document, TokenizedDocument
from sea.tokenizer cimport Tokenizer
from libc.stdint cimport uint8_t, uint64_t, uint32_t, int64_t
from libcpp.utility cimport pair
from libcpp.string cimport string as cstring

cpdef str py_string_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length)
cdef cstring string_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length)

cpdef object py_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length)
cdef Document document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length) noexcept nogil

cpdef object py_tokenized_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, Tokenizer tokenizer)
cdef TokenizedDocument tokenized_document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length, Tokenizer tokenizer) noexcept nogil

cdef class Corpus:
    cdef str save_path
    cdef str data_file_path
    cdef DiskArray disk_array

    cdef object data_file
    cdef uint64_t data_size
    cdef const uint8_t[:] data_map
    cdef uint64_t data_offset

    cpdef object py_get(self, uint64_t idx, object processor)
    cdef Document get_document(self, uint64_t idx) noexcept nogil
    
    cdef pair[uint64_t, uint64_t] _next_line(self) noexcept nogil
    cpdef object py_next(self, object processor)
    cdef pair[int64_t, TokenizedDocument] next_tokenized_document(self, Tokenizer tokenizer) noexcept nogil
    
    cpdef void flush(self)