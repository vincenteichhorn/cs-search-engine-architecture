import math


cpdef float fielded_bm25(object field_token_frequencies, object field_lengths, object average_field_lengths, int df, int N, float K1, list B1s, list boosts):

    cdef float tf = 0.0
    for (tff, bf, lf, alf, boost) in zip(field_token_frequencies, B1s, field_lengths, average_field_lengths, boosts):
        tf += boost * (tff)/(1 - bf + bf * lf/alf)
    
    cdef float idf = math.log((N - df + 0.5)/(df + 0.5))

    cdef float bm25 = idf * (tf * (K1 + 1))/(tf + K1)
    if bm25 < 0.0:
        return 0.0
    return bm25