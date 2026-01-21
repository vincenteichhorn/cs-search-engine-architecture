from libcpp.vector cimport vector
from sea.document cimport SearchResultPosting
from libc.stdint cimport uint32_t
from libcpp cimport bool as cbool
from libc.stdlib cimport abs as cabs

cdef SearchResultPosting merge_postings(SearchResultPosting& posting1, SearchResultPosting& posting2) noexcept nogil:
    assert posting1.doc_id == posting2.doc_id
    cdef SearchResultPosting merged_posting = posting1
    cdef uint32_t i
    for i in range(posting2.tokens.size()):
        merged_posting.tokens.push_back(posting2.tokens[i])
    for i in range(posting2.scores.size()):
        merged_posting.scores.push_back(posting2.scores[i])
    merged_posting.total_score += posting2.total_score
    for i in range(posting2.field_frequencies.size()):
        merged_posting.field_frequencies.push_back(posting2.field_frequencies[i])
    for i in range(posting2.char_positions.size()):
        merged_posting.char_positions.push_back(posting2.char_positions[i])
    merged_posting.snippet_position = posting1.snippet_position
    return merged_posting

cdef cbool phrase_constraint(SearchResultPosting& posting1, SearchResultPosting& posting2, uint32_t k) noexcept nogil:
    cdef uint32_t i = 0
    cdef uint32_t j = 0
    cdef uint32_t l1 = posting1.char_positions.size() - 1
    cdef uint32_t l2 = posting2.char_positions.size() - 1
    cdef uint32_t n = posting1.char_positions[l1].size()
    cdef uint32_t m = posting2.char_positions[l2].size()
    cdef uint32_t abs_dist

    while i < n and j < m:
        abs_dist = posting2.char_positions[l2][j] - posting1.char_positions[l1][i] 
        # distance from posting1 to posting2, so that the order is: posting1 before posting2
        if abs_dist < 0:
            abs_dist = k+1  # if negative the order is wrong, set to large value
        if abs_dist <= k:
            posting1.snippet_position = posting1.char_positions[l1][i]
            return True
        elif posting1.char_positions[l1][i] + k < posting2.char_positions[l2][j]:
            i += 1
        else:
            j += 1
    return False

cdef void intersection(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil:

    cdef vector[SearchResultPosting] new_items = vector[SearchResultPosting]()
    cdef uint32_t num_new_items = self_items.size()
    if other_items.size() < num_new_items:
        num_new_items = other_items.size()
    new_items.reserve(num_new_items)

    cdef int i = 0
    cdef int j = 0
    cdef int n = self_items.size()
    cdef int m = other_items.size()
    cdef SearchResultPosting posting1, posting2

    while i < n and j < m:
        posting1 = self_items[i]
        posting2 = other_items[j]

        if posting1.doc_id < posting2.doc_id:
            # advance self;
            i += 1
        elif posting1.doc_id > posting2.doc_id:
            # advance other;
            j += 1
        else:
            # equal 
            new_items.push_back(merge_postings(posting1, posting2))
            i += 1
            j += 1
    self_items.swap(new_items)

cdef void intersection_phrase(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items, uint32_t k) noexcept nogil:

    cdef vector[SearchResultPosting] new_items = vector[SearchResultPosting]()
    cdef uint32_t num_new_items = self_items.size()
    if other_items.size() < num_new_items:
        num_new_items = other_items.size()
    new_items.reserve(num_new_items)

    cdef int i = 0
    cdef int j = 0
    cdef int n = self_items.size()
    cdef int m = other_items.size()
    cdef SearchResultPosting posting1, posting2

    while i < n and j < m:
        posting1 = self_items[i]
        posting2 = other_items[j]

        if posting1.doc_id < posting2.doc_id:
            # advance self;
            i += 1
        elif posting1.doc_id > posting2.doc_id:
            # advance other;
            j += 1
        else:
            # equal 
            if phrase_constraint(posting1, posting2, k):
                new_items.push_back(merge_postings(posting1, posting2))
            i += 1
            j += 1
    self_items.swap(new_items)

cdef void union(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil:

    cdef vector[SearchResultPosting] new_items = vector[SearchResultPosting]()
    cdef uint32_t num_new_items = self_items.size() + other_items.size()
    new_items.reserve(num_new_items)
    cdef int i = 0
    cdef int j = 0
    cdef int n = self_items.size()
    cdef int m = other_items.size()
    cdef SearchResultPosting posting1, posting2

    while i < n and j < m:
        posting1 = self_items[i]
        posting2 = other_items[j]

        if posting1.doc_id < posting2.doc_id:
            # advance self; 
            new_items.push_back(posting1)
            i += 1
        elif posting1.doc_id > posting2.doc_id:
            # advance other;
            new_items.push_back(posting2)
            j += 1
        else:
            # equal doc_ids
            new_items.push_back(merge_postings(posting1, posting2))
            i += 1
            j += 1

    while i < n:
        new_items.push_back(self_items[i])
        i += 1

    while j < m:
        new_items.push_back(other_items[j])
        j += 1

    self_items.swap(new_items)

cdef void difference(vector[SearchResultPosting]& self_items, vector[SearchResultPosting]& other_items) noexcept nogil:

    cdef vector[SearchResultPosting] new_items = vector[SearchResultPosting]()
    cdef uint32_t num_new_items = self_items.size()
    new_items.reserve(num_new_items)
    cdef int i = 0
    cdef int j = 0
    cdef int n = self_items.size()
    cdef int m = other_items.size()
    cdef SearchResultPosting posting1, posting2

    while i < n and j < m:
        posting1 = self_items[i]
        posting2 = other_items[j]

        if posting1.doc_id == posting2.doc_id:
            # skip both; 
            i += 1
            j += 1
        elif posting1.doc_id < posting2.doc_id:
            # keep posting1
            new_items.push_back(posting1)
            i += 1
        else:
            # advance other;
            j += 1

    while i < n:
        new_items.push_back(self_items[i])
        i += 1

    self_items.swap(new_items)

