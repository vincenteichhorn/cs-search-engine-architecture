from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int32_t
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.string cimport string as cstring
from Stemmer import Stemmer

cdef list STOPWORDS = [
    b"a", b"an", b"and", b"are", b"as", b"at", b"be", b"by", b"can", b"for", b"from",
    b"have", b"if", b"in", b"is", b"it", b"may", b"not", b"of", b"on", b"or", b"tbd",
    b"that", b"the", b"this", b"to", b"us", b"we", b"when", b"will", b"with",
    b"yet", b"you", b"your"
]
cdef list QUERY_EXCLUDE_WORDS = [
    b"and", b"or", b"not"
]

cdef class Tokenizer:
    
    def __cinit__(self, str save_path):
        
        self.save_path = save_path
        self.stopwords = cset[cstring]()
        self.query_stopwords = cset[cstring]()
        cdef bytes word
        for word in STOPWORDS:
            self.stopwords.insert(cstring(word))
        for word in QUERY_EXCLUDE_WORDS:
            self.query_stopwords.insert(cstring(word))

        self.stemmer = Stemmer("english")
        self.disk_array = DiskArray(save_path, name="tokenizer")
        self.vocabulary = unordered_map[cstring, uint32_t]()
        self.max_token_id = self.disk_array.current_idx
        
    cdef inline bint _is_alphanum(self, char c):
        return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9')
    
    cdef inline char _to_lower(self, char c):
        if c >= 'A' and c <= 'Z':
            return c | 0x20
        return c

    cdef inline cstring _ascii_lower(self, cstring s):
        cdef uint64_t n = s.size()
        # construct a string of length n and write chars in place (no malloc/free)
        cdef cstring ret = cstring('\0', n)
        cdef uint64_t i
        for i in range(n):
            ret[i] = self._to_lower(s[i])
        return ret

    cdef inline bint _is_stopword(self, cstring word, bint is_query):
        if is_query:
            return self.query_stopwords.find(word) != self.query_stopwords.end()
        else:
            return self.stopwords.find(word) != self.stopwords.end()

    cdef void _scan(self, cstring text, vector[cstring]& tokens, vector[uint32_t]& char_positions, bint is_query):
        cdef int32_t i = 0, start = -1, n = <uint32_t>text.size()
        cdef char c
        cdef cstring lowered = self._ascii_lower(text)
        cdef cstring token

        while i < n:
            c = lowered[i]
            if self._is_alphanum(c):
                if start < 0:
                    start = i
            else:
                if start >= 0:
                    token = lowered.substr(start, i - start)
                    if not self._is_stopword(token, is_query):
                        tokens.push_back(token)
                        char_positions.push_back(<uint32_t>start)
                    start = -1
                if is_query and (c == '"' or c == '(' or c == ')'):
                    tokens.push_back(cstring(&c, 1))
                    char_positions.push_back(<uint32_t>i)
            i += 1

        # process the last token at end of string
        if start >= 0:
            token = lowered.substr(start, n - start)
            if not self._is_stopword(token, is_query):
                tokens.push_back(token)
                char_positions.push_back(<uint32_t>start)

    cdef TokenizedField tokenize(self, bytes text, bint is_query):
        cdef cstring ctext = cstring(text)

        cdef TokenizedField result
        result.char_positions = vector[uint32_t]()
        cdef vector[cstring] temp_tokens = vector[cstring]()
        self._scan(ctext, temp_tokens, result.char_positions, is_query)
        result.tokens = vector[uint32_t](temp_tokens.size())
        result.length = temp_tokens.size()

        cdef bint stemming = True

        cdef uint32_t i = 0
        cdef cstring token_str
        cdef bytes stemmed_token
        cdef const uint8_t* ptr
        cdef uint32_t token_id
        for i in range(result.length):
            if stemming:
                stemmed_token = self.stemmer.stemWord(<const char*>temp_tokens[i].data())
                token_str = cstring(stemmed_token)
            else:
                token_str = temp_tokens[i]
            if self.vocabulary.find(token_str) == self.vocabulary.end():
                self.vocabulary[token_str] = self.max_token_id
                token_id = self.max_token_id
                self.max_token_id += 1
                ptr = <const uint8_t*>token_str.data()
                self.disk_array.append(ptr, 0, <uint64_t>token_str.length())
            else:
                token_id = self.vocabulary[token_str]
            result.tokens[i] = token_id

        return result

    cpdef tuple py_tokenize(self, bytes text, bint is_query):
        cdef list tokens = []
        cdef list char_positions = []
        cdef TokenizedField result = self.tokenize(text, is_query)
        cdef vector[uint32_t] c_tokens = result.tokens
        cdef vector[uint32_t] c_char_positions = result.char_positions
        cdef uint32_t n = result.length
        
        cdef uint32_t i = 0
        for i in range(n):
            tokens.append(self.disk_array.get(c_tokens[i]).to_bytearray().decode('utf-8'))
            char_positions.append(c_char_positions[i])

        return tokens, char_positions