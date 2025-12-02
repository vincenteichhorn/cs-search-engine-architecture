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

    def __cinit__(self, str algorithm="english"):
        self.cobj = sb_stemmer_new(algorithm.encode('ascii'), b"UTF_8")
        if self.cobj == NULL:
            raise ValueError(f"Algorithm {algorithm} not found")

    def __dealloc__(self):
        if self.cobj != NULL:
            sb_stemmer_delete(self.cobj)

    cdef uint32_t stem(self, unsigned char* word, uint32_t length) noexcept nogil:
        """Stem a UTF-8 encoded word (bytes) without Python GIL overhead."""
        c_word = sb_stemmer_stem(self.cobj, word, length)
        length = sb_stemmer_length(self.cobj)
        return length
    
    cpdef bytes py_stem(self, bytes word):
        """Stem a UTF-8 encoded word (bytes)."""
        cdef unsigned char* c_word = <unsigned char*>word
        cdef uint32_t length = len(word)
        cdef uint32_t stemmed_length = self.stem(c_word, length)
        return bytes(<char*>c_word)[:stemmed_length]