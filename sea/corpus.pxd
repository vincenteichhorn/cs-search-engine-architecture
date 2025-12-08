from sea.util.disk_array cimport DiskArray
from sea.document cimport Document, TokenizedDocument
from sea.tokenizer cimport Tokenizer
from libc.stdint cimport uint8_t, uint64_t, uint32_t, int64_t
from libcpp.utility cimport pair
from libcpp.string cimport string as cstring
from libc.stdio cimport fdopen, fgets, fclose, FILE
from libc.string cimport strlen


cpdef str py_string_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length)
cdef cstring string_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length) noexcept nogil

cpdef object doc_to_dict(Document doc)
cpdef object py_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, bint lowercase)
cdef Document document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length, bint lowercase) noexcept nogil

cpdef object py_tokenized_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, Tokenizer tokenizer)
cdef TokenizedDocument tokenized_document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length, Tokenizer tokenizer) noexcept nogil

ctypedef uint8_t* BytePtr

cdef class Corpus:
    cdef str save_path
    cdef bint serve
    cdef str data_file_path
    cdef DiskArray disk_array

    cdef int data_file_fd
    cdef FILE* data_file_ptr
    cdef size_t data_size
    cdef uint64_t data_offset
    cdef size_t max_line_length
    cdef BytePtr line_buffer

    cpdef object py_get(self, uint64_t idx, object processor)
    cdef Document get_document(self, uint64_t idx, bint lowercase) noexcept nogil
    
    cdef pair[BytePtr, uint64_t] _next_line(self) noexcept nogil
    cpdef object py_next(self, object processor)
    cdef TokenizedDocument next_tokenized_document(self, Tokenizer tokenizer) noexcept nogil
    
    cpdef void flush(self)
    cdef void _flush(self) noexcept nogil
