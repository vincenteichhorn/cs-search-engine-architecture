from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from sea.tokenizer cimport Tokenizer, fnv1a_hash
from libcpp.string cimport string as cstring
from cython.operator cimport preincrement, dereference
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libcpp.utility cimport pair


cdef class SpellingCorrector:

    def __cinit__(self, Tokenizer tokenizer, token_freq_map, exclude_threshold=100):
        self.tokenizer = tokenizer
        self.token_freq_map = token_freq_map
        self.k = 2
        self.exclude_threshold = exclude_threshold
        self.kgram_index = self._build_kgram_index()

    cdef unordered_map[uint64_t, vector[uint64_t]] _build_kgram_index(self) noexcept nogil:
        cdef unordered_map[uint64_t, vector[uint64_t]] kgram_index
        cdef unordered_map[uint64_t, uint32_t].iterator it = self.token_freq_map.begin()
        cdef uint64_t token_id
        cdef uint64_t hash
        cdef cstring token
        cdef char* token_bytes
        cdef size_t i, length
        cdef vector[uint64_t] bigram_hashes
        while it != self.token_freq_map.end():
            if dereference(it).second < self.exclude_threshold:
                preincrement(it)
                continue
            token_id = dereference(it).first
            token = self.tokenizer.get(token_id)
            bigram_hashes = self._get_bigram_hashes(token)
            for i in range(bigram_hashes.size()):
                hash = bigram_hashes[i]
                if kgram_index.find(hash) == kgram_index.end():
                    kgram_index[hash] = vector[uint64_t]()
                kgram_index[hash].push_back(token_id)
            preincrement(it)
        return kgram_index
    
    cdef vector[uint64_t] _get_bigram_hashes(self, cstring token) noexcept nogil:
        cdef vector[uint64_t] bigram_hashes
        cdef char* token_bytes
        cdef size_t i, length
        token_bytes = <char*> malloc(token.size()+2)
        token_bytes[0] = '$'
        memcpy(token_bytes + 1, token.c_str(), token.size())
        token_bytes[token.size()+1] = '$'
        length = token.size() + 2
        for i in range(length - self.k + 1):
            bigram_hashes.push_back(fnv1a_hash(token_bytes + i, self.k))
        free(token_bytes)
        return bigram_hashes

    cdef vector[uint64_t] get_candidates_tokens(self, cstring token) noexcept nogil:
        cdef vector[uint64_t] candidates
        cdef size_t i
        cdef uint64_t hash
        cdef vector[uint64_t] token_ids, bigram_hashes
        bigram_hashes = self._get_bigram_hashes(token)
        for i in range(bigram_hashes.size()):
            hash = bigram_hashes[i]
            if self.kgram_index.find(hash) != self.kgram_index.end():
                token_ids = self.kgram_index[hash]
                candidates.insert(candidates.end(), token_ids.begin(), token_ids.end())
        return candidates
    
    cdef float _jaccard_similarity(self, vector[uint64_t] a, vector[uint64_t] b) noexcept nogil:
        cdef unordered_map[uint64_t, uint32_t] freq_map
        cdef size_t i
        cdef uint32_t intersection = 0
        cdef uint32_t union_count = 0

        for i in range(a.size()):
            freq_map[a[i]] += 1
        for i in range(b.size()):
            if freq_map.find(b[i]) != freq_map.end() and freq_map[b[i]] > 0:
                intersection += 1
                freq_map[b[i]] -= 1
            else:
                freq_map[b[i]] += 1

        cdef unordered_map[uint64_t, uint32_t].iterator it = freq_map.begin()
        while it != freq_map.end():
            union_count += dereference(it).second
            preincrement(it)

        union_count += intersection
        if union_count == 0:
            return 0.0
        return intersection / union_count

    cdef pair[CharPtr, uint32_t] get_top_correction(self, vector[uint64_t] tokens, float min_similarity) noexcept nogil:
        if tokens.size() == 0:
            return pair[CharPtr, uint32_t](NULL, 0)
        
        cdef uint32_t num_corrected = 0
        
        cdef uint64_t i, j
        cdef cstring token, cand, best_cand
        cdef vector[uint64_t] candidate_tokens, candidate_bigrams, token_bigrams
        cdef float similarity, best_similarity
        cdef vector[cstring] corrections
        for i in range(tokens.size()):
            token = self.tokenizer.get(tokens[i])
            token_bigrams = self._get_bigram_hashes(token)
            candidate_tokens = self.get_candidates_tokens(token)
            best_similarity = 0.0
            for j in range(candidate_tokens.size()):
                cand = self.tokenizer.get(candidate_tokens[j])
                candidate_bigrams = self._get_bigram_hashes(cand)
                similarity = self._jaccard_similarity(token_bigrams, candidate_bigrams)
                if similarity > best_similarity and similarity >= min_similarity:
                    best_similarity = similarity
                    best_cand = cand
            if best_similarity > 0.0:
                corrections.push_back(best_cand)
                num_corrected += 1
            else:
                corrections.push_back(token)

        cdef size_t total_length = 0
        for i in range(corrections.size()):
            total_length += corrections[i].size() + 1  # for space or null terminator
        cdef char* result = <char*> malloc(total_length * sizeof(char))
        cdef size_t pos = 0
        for i in range(corrections.size()):
            memcpy(result + pos, corrections[i].c_str(), corrections[i].size())
            pos += corrections[i].size()
            if i < corrections.size() - 1:
                result[pos] = ' '
                pos += 1
            else:
                result[pos] = '\0'
        return pair[CharPtr, uint32_t](result, num_corrected)