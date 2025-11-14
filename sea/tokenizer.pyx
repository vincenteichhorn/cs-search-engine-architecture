import re
import Stemmer

cdef class Tokenizer:
    cdef object stop_words
    cdef object query_stop_words
    cdef object _regex
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
        
        # initialize PyStemmer
        self.stemmer = Stemmer.Stemmer('english')

    cpdef list tokenize(self, str text, bint is_query=False):
        """
        Ultra-fast tokenizer + stemmer using regex + PyStemmer
        """
        cdef list out = []
        cdef object stopset = self.query_stop_words if is_query else self.stop_words
        cdef str token
        cdef str lowered
        cdef str stemmed

        for token in self._regex.findall(text):
            lowered = token.lower()
            if lowered in stopset:
                continue
            stemmed = self.stemmer.stemWord(lowered)
            out.append(stemmed)
        return out

    cpdef list tokenize_document(self, object document):
        cdef list body_toks = self.tokenize(document.body, False)
        cdef list title_toks = self.tokenize(document.title, False)
        body_toks.extend(title_toks)
        return body_toks

    cpdef list tokenize_query(self, str query):
        return self.tokenize(query, True)
