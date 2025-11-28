import re
import Stemmer

cdef class Tokenizer:
    cdef object stop_words
    cdef object query_stop_words
    cdef object _regex
    cdef object _query_regex
    cdef object stemmer

    def __init__(self, stop_words=None):
        if stop_words is None:
            stop_words = (
                "a","an","and","are","as","at","be","by","can","for","from",
                "have","if","in","is","it","may","not","of","on","or","tbd",
                "that","the","this","to","us","we","when","will","with",
                "yet","you","your",
            )
        self.stop_words = set(stop_words)
        self.query_stop_words = set(stop_words) - {"and", "or", "not"}
        
        # precompile regex for word tokenization
        self._regex = re.compile(r'\b\w+\b')
        self._query_regex = re.compile(r'"|\(|\)|\w+')
        
        # initialize PyStemmer
        self.stemmer = Stemmer.Stemmer('english')

    cpdef tuple tokenize(self, str text, bint is_query=False):
        """
        Ultra-fast tokenizer + stemmer using regex + PyStemmer
        """
        cdef list out = []
        cdef list char_positions = []
        cdef object stopset = self.query_stop_words if is_query else self.stop_words
        cdef object token
        cdef str lowered
        cdef str stemmed

        cdef object regex = self._query_regex if is_query else self._regex

        for token in regex.finditer(text):
            lowered = token.group().lower()
            if lowered in stopset or not lowered.isascii():
                continue
            stemmed = self.stemmer.stemWord(lowered)
            out.append(stemmed)
            char_positions.append(token.start())
        return out, char_positions

    cpdef tuple tokenize_document(self, object document):
        cdef list title_toks
        cdef list char_title_positions
        title_toks, char_title_positions = self.tokenize(document.title, False)
        cdef int num_title_toks = len(title_toks)
        cdef int num_title_chars = len(document.title)
        cdef list body_toks
        cdef list char_body_positions
        body_toks, char_body_positions = self.tokenize(document.body, False)
        title_toks.extend(body_toks)
        char_title_positions.extend(char_body_positions)
        return title_toks, char_title_positions, num_title_chars, num_title_toks

    cpdef list tokenize_query(self, str query):
        cdef list tokens
        cdef list char_positions
        tokens, char_positions = self.tokenize(query, True)
        return tokens