from libc.stdint cimport uint32_t

cdef extern from "libstemmer.h":
    ctypedef unsigned char sb_symbol
    cdef struct sb_stemmer # type: ignore
    cdef char ** sb_stemmer_list() nogil
    cdef sb_stemmer* sb_stemmer_new(const char* algorithm, const char* charenc) nogil 
    cdef const sb_symbol* sb_stemmer_stem(sb_stemmer* stemmer, const sb_symbol* word, int size) nogil
    cdef void sb_stemmer_delete(sb_stemmer* stemmer) nogil
    cdef int sb_stemmer_length(sb_stemmer* stemmer) nogil

cdef class FastStemmer:
    cdef sb_stemmer* cobj

    cdef uint32_t stem(self, unsigned char* word, uint32_t length) noexcept nogil
    cpdef bytes py_stem(self, bytes word)