
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy
from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cython.operator cimport dereference
from libcpp cimport bool as cbool
from sea.document cimport Posting
from sea.util.memory cimport read_uint32

cdef cbool compare_postings_ptr(const Posting* a, const Posting* b) noexcept nogil:
    return dereference(a).doc_id < dereference(b).doc_id

cdef void free_document(Document* doc) noexcept nogil:
    if doc.url != NULL:
        free(doc.url)
        doc.url = NULL
    if doc.title != NULL:
        free(doc.title)
        doc.title = NULL
    if doc.body != NULL:
        free(doc.body)
        doc.body = NULL
    if doc.snippet != NULL:
        free(doc.snippet)
        doc.snippet = NULL

cdef void free_posting(Posting* posting, cbool all) noexcept nogil:
    if posting.field_frequencies != NULL:
        free(posting.field_frequencies)
        posting.field_frequencies = NULL
    if posting.field_lengths != NULL and all:
        free(posting.field_lengths)
        posting.field_lengths = NULL
    posting.char_positions.clear()
    # posting.token_positions.clear()

cdef void free_tokenized_document(TokenizedDocument* tokenized_doc) noexcept nogil:
    cdef uint32_t i
    if tokenized_doc.field_lengths != NULL:
        free(tokenized_doc.field_lengths)
        tokenized_doc.field_lengths = NULL
    tokenized_doc.tokens.clear()

cdef void free_tokenized_document_with_postings(TokenizedDocument* tokenized_doc) noexcept nogil:
    cdef uint32_t i
    if tokenized_doc.field_lengths != NULL:
        free(tokenized_doc.field_lengths)
        tokenized_doc.field_lengths = NULL
    for i in range(tokenized_doc.postings.size()):
        free_posting(&tokenized_doc.postings[i], True)
    tokenized_doc.postings.clear()

cdef pair[BytePtr, uint32_t] serialize_postings(vector[Posting*]* postings) noexcept nogil:

    cdef uint32_t total_size
    cdef uint32_t current_size = 0
    cdef uint32_t current_capacity = 1024
    cdef uint8_t* buffer = <uint8_t*>malloc(current_capacity)
    cdef uint8_t* cur_ptr = <uint8_t*>buffer
    cdef uint32_t num_positions
    cdef uint32_t num_fields
    cdef uint32_t i, j
    cdef Posting* posting

    total_size = 0

    for i in range(dereference(postings).size()):
        posting = dereference(postings)[i]
        num_positions = posting.char_positions.size()
        num_fields = posting.num_fields

        current_size = sizeof(uint32_t) + sizeof(float) + sizeof(uint32_t) * (1 + num_positions) + sizeof(uint32_t) * (1 + num_fields) + sizeof(uint32_t) * num_fields
        total_size += current_size
        if total_size > current_capacity:
            current_capacity = total_size
            buffer = <uint8_t*>realloc(buffer, current_capacity)
            cur_ptr = buffer + total_size - current_size

        memcpy(cur_ptr, &posting.doc_id, sizeof(uint32_t))
        cur_ptr += sizeof(uint32_t)
        memcpy(cur_ptr, &posting.score, sizeof(float))
        cur_ptr += sizeof(float)

        memcpy(cur_ptr, &num_fields, sizeof(uint32_t))
        cur_ptr += sizeof(uint32_t)
        for j in range(posting.num_fields):
            memcpy(cur_ptr, &posting.field_frequencies[j], sizeof(uint32_t))
            cur_ptr += sizeof(uint32_t)

        for j in range(posting.num_fields):
            memcpy(cur_ptr, &posting.field_lengths[j], sizeof(uint32_t))
            cur_ptr += sizeof(uint32_t)

        memcpy(cur_ptr, &num_positions, sizeof(uint32_t))
        cur_ptr += sizeof(uint32_t)
        for j in range(posting.char_positions.size()):
            memcpy(cur_ptr, &posting.char_positions[j], sizeof(uint32_t))
            cur_ptr += sizeof(uint32_t)

    return pair[BytePtr, uint32_t](buffer, total_size)

cdef uint32_t get_posting_list_length(const uint8_t* data, uint32_t length) noexcept nogil:
    cdef uint32_t cur = 0
    cdef uint32_t count = 0
    cdef uint32_t num_fields
    cdef uint32_t num_positions
    while cur < length:
        cur += sizeof(uint32_t) + sizeof(float) # doc_id + score
        memcpy(&num_fields, data + cur, sizeof(uint32_t))
        cur += sizeof(uint32_t)
        cur += num_fields * sizeof(uint32_t) # field_frequencies
        cur += num_fields * sizeof(uint32_t) # field_lengths
        memcpy(&num_positions, data + cur, sizeof(uint32_t))
        cur += sizeof(uint32_t)
        cur += num_positions * sizeof(uint32_t) # char_positions
        count += 1
    return count

cdef vector[Posting] deserialize_postings(const uint8_t* data, uint32_t length) noexcept nogil:
    cdef vector[Posting] postings = vector[Posting]()

    cdef uint32_t i, j
    cdef Posting posting
    cdef uint32_t num_positions
    cdef uint32_t pos
    cdef uint32_t cur = 0
    # cdef uint64_t last_doc_id = 0

    while cur < length:
        memcpy(&posting.doc_id, data + cur, sizeof(uint32_t))
        # posting.doc_id += last_doc_id
        # last_doc_id = posting.doc_id

        cur += sizeof(uint32_t)
        memcpy(&posting.score, data + cur, sizeof(float))
        cur += sizeof(float)

        memcpy(&posting.num_fields, data + cur, sizeof(uint32_t))
        cur += sizeof(uint32_t)
        posting.field_frequencies = <uint32_t*>malloc(posting.num_fields * sizeof(uint32_t))
        for j in range(posting.num_fields):
            memcpy(&posting.field_frequencies[j], data + cur, sizeof(uint32_t))
            cur += sizeof(uint32_t)
        
        posting.field_lengths = <uint32_t*>malloc(posting.num_fields * sizeof(uint32_t))
        for j in range(posting.num_fields):
            memcpy(&posting.field_lengths[j], data + cur, sizeof(uint32_t))
            cur += sizeof(uint32_t)

        memcpy(&num_positions, data + cur, sizeof(uint32_t))
        cur += sizeof(uint32_t)
        posting.char_positions = vector[uint32_t]()
        for j in range(num_positions):
            memcpy(&pos, data + cur, sizeof(uint32_t))
            posting.char_positions.push_back(pos)
            cur += sizeof(uint32_t)

        postings.push_back(posting)

    return postings

cdef vector[SearchResultPosting] deserialize_search_result_postings(const uint8_t* data, uint32_t length, uint32_t token_id) noexcept nogil:
    cdef vector[SearchResultPosting] postings = vector[SearchResultPosting]()

    cdef uint32_t i, j
    cdef SearchResultPosting sr_posting
    cdef uint32_t num_positions
    cdef uint32_t cur = 0

    while cur < length:
        memcpy(&sr_posting.doc_id, data + cur, sizeof(uint32_t))

        sr_posting.tokens = vector[uint64_t]()
        sr_posting.tokens.push_back(token_id)

        cur += sizeof(uint32_t)
        memcpy(&sr_posting.total_score, data + cur, sizeof(float))
        sr_posting.scores = vector[float]()
        sr_posting.scores.push_back(sr_posting.total_score)
        cur += sizeof(float)

        memcpy(&sr_posting.num_fields, data + cur, sizeof(uint32_t))
        cur += sizeof(uint32_t)
        sr_posting.field_frequencies = vector[vector[uint32_t]]()
        sr_posting.field_frequencies.push_back(vector[uint32_t](sr_posting.num_fields))
        for j in range(sr_posting.num_fields):
            memcpy(&sr_posting.field_frequencies[0][j], data + cur, sizeof(uint32_t))
            cur += sizeof(uint32_t)
        
        sr_posting.field_lengths = vector[uint32_t](sr_posting.num_fields)
        for j in range(sr_posting.num_fields):
            memcpy(&sr_posting.field_lengths[j], data + cur, sizeof(uint32_t))
            cur += sizeof(uint32_t)

        memcpy(&num_positions, data + cur, sizeof(uint32_t))
        cur += sizeof(uint32_t)
        sr_posting.char_positions = vector[vector[uint32_t]]()
        sr_posting.char_positions.push_back(vector[uint32_t](num_positions))
        for j in range(num_positions):
            memcpy(&sr_posting.char_positions[0][j], data + cur, sizeof(uint32_t))
            cur += sizeof(uint32_t)
        sr_posting.snippet_position = 0

        postings.push_back(sr_posting)

    return postings

cdef vector[SearchResultPosting] create_search_result_postings(vector[Posting]& postings, uint64_t token) noexcept nogil:
    cdef vector[SearchResultPosting] result_postings = vector[SearchResultPosting]()
    cdef uint32_t i, j
    cdef SearchResultPosting sr_posting
    cdef Posting posting

    for i in range(postings.size()):
        posting = postings[i]
        
        sr_posting.doc_id = posting.doc_id
        sr_posting.num_fields = posting.num_fields

        sr_posting.tokens = vector[uint64_t]()
        sr_posting.tokens.push_back(token)

        sr_posting.scores = vector[float]()
        sr_posting.scores.push_back(posting.score)
        sr_posting.total_score = posting.score

        sr_posting.field_frequencies = vector[vector[uint32_t]]()
        sr_posting.field_frequencies.push_back(vector[uint32_t]())
        for j in range(posting.num_fields):
            sr_posting.field_frequencies[0].push_back(posting.field_frequencies[j])
        sr_posting.field_lengths = vector[uint32_t]()
        for j in range(posting.num_fields):
            sr_posting.field_lengths.push_back(posting.field_lengths[j])
        
        sr_posting.char_positions = vector[vector[uint32_t]]()
        sr_posting.char_positions.push_back(vector[uint32_t]())
        for j in range(posting.char_positions.size()):
            sr_posting.char_positions[0].push_back(posting.char_positions[j])
        
        free_posting(&posting, True)
        sr_posting.similarity_score = 0.0
        result_postings.push_back(sr_posting)
    return result_postings

cdef pair[float, size_t] update_posting_score(const uint8_t* data, size_t offset, float idf, float bm25k, vector[float]& field_boosts, vector[float]& bm25_bs, vector[float]& avg_field_lengths) noexcept nogil:
    


    cdef uint32_t i
    cdef size_t cur = offset

    cur += sizeof(uint32_t) + sizeof(float) # skip doc_id and old score
    cdef uint32_t num_fields = read_uint32(data, cur)
    cur += sizeof(uint32_t)
    
    cdef float tf = 0.0
    cdef uint32_t field_length
    cdef uint32_t field_frequency
    for i in range(num_fields):
        field_frequency = read_uint32(data, cur + i * sizeof(uint32_t))
        field_length = read_uint32(data, cur + (num_fields + i) * sizeof(uint32_t))
        tf += (field_frequency * field_boosts[i]) / (1.0 + bm25_bs[i] + (1.0 - bm25_bs[i]) * (field_length / avg_field_lengths[i]))
    
    cdef float score = idf * ((tf * (bm25k + 1.0)) / (tf + bm25k))

    memcpy(<void*>(<uint8_t*>data + offset + sizeof(uint32_t)), &score, sizeof(float))

    cdef uint32_t num_positions = read_uint32(data, cur + num_fields * 2 * sizeof(uint32_t))
    cdef size_t size_of_posting = sizeof(uint32_t) + sizeof(float) + sizeof(uint32_t) * (1 + num_positions) + sizeof(uint32_t) * (1 + num_fields) + sizeof(uint32_t) * num_fields
    
    return pair[float, size_t](score, size_of_posting)