from libcpp.vector cimport vector
from sea.document cimport Posting
from libc.stdint cimport uint32_t
from libcpp cimport bool as cbool

cdef Posting merge_postings(Posting posting1, Posting posting2) noexcept nogil:
    assert posting1.doc_id == posting2.doc_id
    cdef Posting merged_posting
    merged_posting.doc_id = posting1.doc_id
    merged_posting.score = posting1.score + posting2.score
    merged_posting.num_fields = posting2.num_fields
    merged_posting.field_frequencies = posting2.field_frequencies
    merged_posting.field_lengths = posting2.field_lengths
    merged_posting.char_positions = posting2.char_positions
    return merged_posting

cdef cbool phrase_constraint(Posting posting1, Posting posting2, uint32_t dist) noexcept nogil:
    cdef uint32_t i = 0
    cdef uint32_t j = 0
    cdef uint32_t n = posting1.char_positions.size()
    cdef uint32_t m = posting2.char_positions.size()

    while i < n and j < m:
        if posting1.char_positions[i] + dist == posting2.char_positions[j]:
            return True
        elif posting1.char_positions[i] + dist < posting2.char_positions[j]:
            i += 1
        else:
            j += 1
    return False

cdef void intersection(vector[Posting]& self_items, vector[Posting]& other_items, cbool phrase) noexcept nogil:

    cdef vector[Posting] new_items = vector[Posting]()
    cdef uint32_t num_new_items = self_items.size()
    if other_items.size() < num_new_items:
        num_new_items = other_items.size()
    new_items.reserve(num_new_items)

    cdef int i = 0
    cdef int j = 0
    cdef int n = self_items.size()
    cdef int m = other_items.size()
    cdef Posting posting1, posting2

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
            if phrase and phrase_constraint(posting1, posting2, 1):
                new_items.push_back(merge_postings(posting1, posting2))
            elif not phrase:
                new_items.push_back(merge_postings(posting1, posting2))
            i += 1
            j += 1
    self_items.swap(new_items)

cdef void union(vector[Posting]& self_items, vector[Posting]& other_items) noexcept nogil:

    cdef vector[Posting] new_items = vector[Posting]()
    cdef uint32_t num_new_items = self_items.size() + other_items.size()
    new_items.reserve(num_new_items)
    cdef int i = 0
    cdef int j = 0
    cdef int n = self_items.size()
    cdef int m = other_items.size()
    cdef Posting posting1, posting2

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

cdef void difference(vector[Posting]& self_items, vector[Posting]& other_items) noexcept nogil:

    cdef vector[Posting] new_items = vector[Posting]()
    cdef uint32_t num_new_items = self_items.size()
    new_items.reserve(num_new_items)
    cdef int i = 0
    cdef int j = 0
    cdef int n = self_items.size()
    cdef int m = other_items.size()
    cdef Posting posting1, posting2

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

