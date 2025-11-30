import Stemmer

cdef class Tokenizer:
    cdef set stop_words
    cdef set query_stop_words
    cdef object stemmer

    def __init__(self, stop_words=None):
        if stop_words is None:
            stop_words = (
                "a","an","and","are","as","at","be","by","can","for","from",
                "have","if","in","is","it","may","not","of","on","or","tbd",
                "that","the","this","to","us","we","when","will","with",
                "yet","you","your",
            )

        self.stop_words = {intern(w) for w in stop_words}
        self.query_stop_words = self.stop_words - {intern("and"), intern("or"), intern("not")}

        self.stemmer = Stemmer.Stemmer('english')

    cdef inline bint _is_word_char(self, char c):
        return (c >= '0' and c <= '9') or \
               (c >= 'A' and c <= 'Z') or \
               (c >= 'a' and c <= 'z') or \
               c == '_'

    cdef inline str _ascii_lower(self, str s):
        return ''.join(chr(ord(c) | 0x20) if 'A' <= c <= 'Z' else c for c in s)

    cdef _scan_normal(self, str text, list temp_tokens, list temp_positions):
        cdef Py_ssize_t i = 0, start = -1, n = len(text)
        cdef char c
        while i < n:
            c = text[i]
            if self._is_word_char(c):
                if start < 0:
                    start = i
            else:
                if start >= 0:
                    temp_tokens.append(text[start:i])
                    temp_positions.append(start)
                    start = -1
            i += 1
        if start >= 0:
            temp_tokens.append(text[start:n])
            temp_positions.append(start)

    cdef _scan_query(self, str text, list temp_tokens, list temp_positions):
        cdef Py_ssize_t i = 0, start = -1, n = len(text)
        cdef char c
        while i < n:
            c = text[i]
            if self._is_word_char(c):
                if start < 0:
                    start = i
            else:
                if start >= 0:
                    temp_tokens.append(text[start:i])
                    temp_positions.append(start)
                    start = -1
                # Keep quotes and parentheses as tokens
                if c == '"' or c == '(' or c == ')':
                    temp_tokens.append(text[i:i+1])
                    temp_positions.append(i)
            i += 1
        if start >= 0:
            temp_tokens.append(text[start:n])
            temp_positions.append(start)

    cpdef tuple tokenize(self, str text, bint is_query=False):
        """
        Tokenize text into stemmed words, remove stopwords, track positions.
        Returns: (tokens_list, char_positions_list)
        """
        cdef list temp_tokens = []
        cdef list temp_positions = []

        # Scan text according to mode
        if is_query:
            self._scan_query(text, temp_tokens, temp_positions)
        else:
            self._scan_normal(text, temp_tokens, temp_positions)

        # Filter stopwords and lowercase
        cdef list filtered_tokens = []
        cdef list filtered_positions = []
        cdef Py_ssize_t i, n = len(temp_tokens)
        cdef str t, lowered
        cdef set stopset = self.query_stop_words if is_query else self.stop_words

        for i in range(n):
            t = temp_tokens[i]
            lowered = intern(self._ascii_lower(t))
            if lowered in stopset or not lowered.isascii():
                continue
            filtered_tokens.append(lowered)
            filtered_positions.append(temp_positions[i])

        # Batch stem
        cdef list stemmed_tokens = self.stemmer.stemWords(filtered_tokens)

        return stemmed_tokens, filtered_positions
