from sea.posting_list import PostingList


cpdef list get_bigrams(str term):
    cdef list results = []

    if len(term) == 0:
        return results

    for i in range(len(term)):
        if i - 1 < 0:
            results.append("$")
        results[-1] += term[i]
        results.append(term[i])
    results[-1] += "$"

    return results

cdef float jaccard_similarity(list a, list b):
    cdef set set_a = set(a)
    cdef set set_b = set(b)
    cdef int num_intersection = len(set_a & set_b)
    cdef int num_union = len(set_a | set_b)
    cdef float similarity = num_intersection / num_union
    return similarity


cdef identity(x):
    return x

cdef class SpellingCorrector:

    cdef object index

    def __cinit__(self, list tokens, str alphabet = None):
        """
        Constructs a bigram index for spelling correction

        Args:
            list of distict tokens

        """
        
        cdef list bigrams
        self.index = {}
        if alphabet is None:
            alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        alphabet += "$"
        for a in alphabet:
            for b in alphabet:
                self.index[f"{a}{b}"] = PostingList(identity)

        for token in tokens:
            bigrams = get_bigrams(token)
            for bi in bigrams:
                if bi not in self.index:
                    self.index[bi] = PostingList(identity)
                self.index[bi].add(token)

    cpdef object get_candidates(self, list bigrams):

        cdef object candidates = PostingList(identity)
        for bi in bigrams:
            candidates.union(self.index[bi])
        return list(candidates)

    cpdef list get_corrections_all(self, str token, float threshold = 0.7):
        
        cdef list bigrams = get_bigrams(token)
        cdef list candidates = self.get_candidates(bigrams)
        cdef list corrections = []

        cdef str cand
        cdef list cand_bigrams
        for cand in candidates:
            cand_bigrams = get_bigrams(cand)
            if jaccard_similarity(bigrams, cand_bigrams) > threshold:
                corrections.append(cand)

        return corrections
    
    cpdef str get_top_correction(self, str token):

        cdef list bigrams = get_bigrams(token)
        cdef list candidates = self.get_candidates(bigrams)
        cdef list corrections = []

        cdef str cand
        cdef list cand_bigrams
        cdef float best_score = 0
        cdef str best_cand
        for cand in candidates:
            cand_bigrams = get_bigrams(cand)
            cand_score = jaccard_similarity(bigrams, cand_bigrams)
            if cand_score >= best_score:
                best_score = cand_score
                best_cand = cand
        return best_cand



