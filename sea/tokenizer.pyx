# cython: boundscheck=False
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int32_t
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.string cimport string as cstring
from sea.util.fast_stemmer cimport FastStemmer
from sea.util.disk_array cimport DiskArray
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC
from libc.stddef cimport size_t
from libcpp.utility cimport pair

ctypedef const char* ccharp

cdef list STOPWORDS = [
    b"a", b"an", b"and", b"are", b"as", b"at", b"be", b"by", b"can", b"for", b"from",
    b"have", b"if", b"in", b"is", b"it", b"may", b"not", b"of", b"on", b"or", b"tbd",
    b"that", b"the", b"this", b"to", b"us", b"we", b"when", b"will", b"with",
    b"yet", b"you", b"your"
]
cdef list QUERY_EXCLUDE_WORDS = [
    b"and", b"or", b"not"
]

cdef uint64_t fnv1a_hash(const char* ptr, size_t length) noexcept nogil:
    cdef uint64_t hash = 14695981039346656037  # FNV offset basis
    cdef uint64_t FNV_PRIME = 1099511628211
    cdef size_t i
    for i in range(length):
        hash ^= <uint8_t>ptr[i]
        hash *= FNV_PRIME  # FNV prime
    return hash

cdef class Tokenizer:
    
    def __cinit__(self, str save_path, bint stem=True):
        
        self.save_path = save_path
        self.stopwords = unordered_set[uint64_t]()
        self.query_stopwords = unordered_set[uint64_t]()
        cdef bytes word
        for word in STOPWORDS:
            self.stopwords.insert(fnv1a_hash(word, len(word)))
        for word in QUERY_EXCLUDE_WORDS:
            self.query_stopwords.insert(fnv1a_hash(word, len(word)))
        self.stem = stem
        if self.stem:
            self.stemmer = FastStemmer()
        self.disk_array = DiskArray(save_path, name="tokenizer")
        self.vocabulary = unordered_map[uint64_t, uint32_t]()
        self.max_token_id = self.disk_array.current_idx
        

    cdef inline bint _is_stopword(self, uint64_t token_hash, bint is_query) noexcept nogil:
        if is_query:
            return self.query_stopwords.find(token_hash) != self.query_stopwords.end()
        else:
            return self.stopwords.find(token_hash) != self.stopwords.end()

    cdef void _scan(self, const char* text, uint32_t length, vector[const char*]& token_ptrs, vector[uint32_t]& token_lens, vector[uint32_t]& char_positions, bint is_query) noexcept nogil:
        cdef int32_t start = -1
        cdef uint32_t i = 0
        while i < length:
            c = text[i] | 0x20  # to_lower inlined
            if (c >= 'a' and c <= 'z') or (c >= '0' and c <= '9'):
                if start < 0:
                    start = i
            else:
                if start >= 0:
                    token_ptrs.push_back(text + start)
                    token_lens.push_back(i - start)
                    char_positions.push_back(start)
                    start = -1
                if is_query and (text[i] == '"' or text[i] == '(' or text[i] == ')'):
                    token_ptrs.push_back(text + i)
                    token_lens.push_back(1)
                    char_positions.push_back(i)
            i += 1
        if start >= 0:
            token_ptrs.push_back(text + start)
            token_lens.push_back(length - start)
            char_positions.push_back(start)

    cdef TokenizedField tokenize(self, const char* text, uint32_t length, bint is_query) noexcept nogil:
        

        cdef TokenizedField result
        cdef vector[const char*] token_ptrs = vector[ccharp]()
        cdef vector[uint32_t] token_lens = vector[uint32_t]()
        cdef vector[uint32_t] char_positions = vector[uint32_t]()

        self._scan(text, length, token_ptrs, token_lens, char_positions, is_query)        
        result.tokens = vector[uint32_t]()
        result.char_positions = vector[uint32_t]()
        result.max_token_id = 0

        cdef uint32_t i = 0
        cdef const char* ptr
        cdef const uint8_t* u_ptr
        cdef uint32_t token_len
        cdef uint32_t token_id
        cdef uint64_t token_hash
        cdef cstring token_str

        for i in range(token_ptrs.size()):
            ptr = token_ptrs[i]
            token_len = token_lens[i]
            if self.stem:
                token_len = self.stemmer.stem(<unsigned char*>ptr, token_len)

            token_hash = fnv1a_hash(ptr, token_len)
            
            if self._is_stopword(token_hash, is_query):
                continue

            if self.vocabulary.find(token_hash) == self.vocabulary.end():
                self.vocabulary[token_hash] = self.max_token_id
                token_id = self.max_token_id
                self.max_token_id += 1
                u_ptr = <const uint8_t*>ptr
                self.disk_array.append(u_ptr, 0, <uint64_t>token_len)
            else:
                token_id = self.vocabulary[token_hash]
            result.tokens.push_back(token_id)
            result.char_positions.push_back(char_positions[i])

            if token_id > result.max_token_id:
                result.max_token_id = token_id
        result.length = result.tokens.size()
        
        return result

    cpdef tuple py_tokenize(self, bytes text, bint is_query):
        cdef list tokens = []
        cdef list char_positions = []
        cdef const char* c_text = text
        cdef TokenizedField result = self.tokenize(c_text, len(text), is_query)
        cdef vector[uint32_t] c_tokens = result.tokens
        cdef vector[uint32_t] c_char_positions = result.char_positions
        cdef uint32_t n = result.length
        
        cdef uint32_t i = 0
        for i in range(n):
            tokens.append(self.disk_array.py_get(c_tokens[i]).decode('utf-8'))
            char_positions.append(c_char_positions[i])

        return tokens, char_positions

    cdef const cstring get(self, uint64_t idx) noexcept nogil:
        if idx > self.max_token_id:
            return cstring()
        cdef pair[const uint8_t*, uint32_t] slice = self.disk_array.get(idx)
        cdef cstring token = cstring(<const char*>slice.first, slice.second)
        return token
    
    cpdef str py_get(self, uint64_t idx):
        cdef cstring token = self.get(idx)
        return token.decode('utf-8')

    cpdef void flush(self):
        self.disk_array.flush()