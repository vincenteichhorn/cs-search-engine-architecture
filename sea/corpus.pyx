# cython: boundscheck=False
from sea.disk_array cimport DiskArray
from sea.document cimport Document, TokenizedDocument, TokenInfo
from sea.tokenizer cimport Tokenizer, TokenizedField
from libc.stdint cimport uint8_t, uint64_t, uint32_t
from libcpp.string cimport string as cstring
from cpython.unicode cimport PyUnicode_DecodeUTF8
from libcpp.vector cimport vector
import os
import mmap
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC

cpdef str identity_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length):
    cdef const uint8_t* ptr = &data[offset]
    return PyUnicode_DecodeUTF8(<const char*>ptr, length, "")

cpdef Document document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length) noexcept nogil:
    cdef const uint8_t* ptr = &data[offset]
    cdef uint32_t start = 0
    cdef int field_index = 0
    cdef char c
    cdef Document doc
    doc.id = id
    cdef uint32_t i = 0
    while i < length:
        c = ptr[i]
        if c == 9 or c == 10 or c == 13: # \t, \n, \r
            # skip field_index 0
            if field_index == 1:
                doc.url = cstring(<const char*>ptr + start, i - start)
            elif field_index == 2:
                doc.title = cstring(<const char*>ptr + start, i - start)
            elif field_index == 3:
                doc.body = cstring(<const char*>ptr + start, i - start)
            start = i + 1
            field_index += 1
        i += 1
    
    # lowercase url, title, body in place
    i = 0
    while i < doc.url.length():
        if doc.url[i] >= 'A' and doc.url[i] <= 'Z':
            doc.url[i] = doc.url[i] | 0x20
        i += 1
    i = 0
    while i < doc.title.length():
        if doc.title[i] >= 'A' and doc.title[i] <= 'Z':
            doc.title[i] = doc.title[i] | 0x20
        i += 1
    i = 0
    while i < doc.body.length():
        if doc.body[i] >= 'A' and doc.body[i] <= 'Z':
            doc.body[i] = doc.body[i] | 0x20
        i += 1

    return doc

cpdef TokenizedDocument tokenized_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, Tokenizer tokenizer) noexcept nogil:
    
    cdef Document doc = document_processor(id, data, offset, length)

    cdef const char* title_ptr = <const char*>doc.title.data()
    cdef const char* body_ptr = <const char*>doc.body.data()
    cdef TokenizedField title_field = tokenizer.tokenize(title_ptr, <uint32_t>doc.title.length(), False)
    cdef TokenizedField body_field = tokenizer.tokenize(body_ptr, <uint32_t>doc.body.length(), False)
    cdef uint32_t max_token_id = title_field.max_token_id
    if body_field.max_token_id > max_token_id:
        max_token_id = body_field.max_token_id

    cdef vector[TokenInfo] token_infos = vector[TokenInfo](max_token_id + 1)
    cdef vector[bint] seen = vector[bint](max_token_id + 1, False)
    cdef uint32_t num_unique_tokens = 0
    cdef uint32_t num_title_tokens = title_field.length
    cdef TokenInfo info
    cdef uint32_t i
    cdef uint32_t token

    for i in range(title_field.length):
        token = title_field.tokens[i]
        if not seen[token]:
            token_infos[token].char_position = title_field.char_positions[i]
            token_infos[token].token_position = i
            token_infos[token].frequency = 1
            seen[token] = True
            num_unique_tokens += 1
        else:
            token_infos[token].frequency += 1
    for i in range(body_field.length):
        token = body_field.tokens[i]
        if not seen[token]:
            token_infos[token].char_position = body_field.char_positions[i]
            token_infos[token].token_position = num_title_tokens + i
            token_infos[token].frequency = 1
            seen[token] = True
            num_unique_tokens += 1
        else:
            token_infos[token].frequency += 1

    cdef TokenizedDocument tdoc
    tdoc.id = doc.id
    tdoc.field_lengths.reserve(2)
    tdoc.field_lengths.push_back(title_field.length)
    tdoc.field_lengths.push_back(body_field.length)
    tdoc.tokens.reserve(num_unique_tokens)
    tdoc.token_infos.reserve(num_unique_tokens)

    i = 0
    while i < seen.size():
        if seen[i]:
            tdoc.tokens.push_back(i)
            tdoc.token_infos.push_back(token_infos[i])
        i += 1

    return tdoc

cdef class Corpus:
    
    def __cinit__(self, save_path, data_file_path):
        self.save_path = save_path
        self.data_file_path = data_file_path
        self.disk_array = DiskArray(save_path, name="corpus")

        self.data_file = open(data_file_path, "rb")
        self.data_size = os.path.getsize(data_file_path)
        self.data_map = mmap.mmap(self.data_file.fileno(), self.data_size, access=mmap.ACCESS_READ)
        self.data_offset = self.disk_array.data_size

    cpdef object get(self, uint64_t idx, object processor):
        cdef const uint8_t[:] buffer = self.disk_array.get(idx)
        cdef const uint8_t* ptr = &buffer[0]
        return processor(idx, ptr, 0, buffer.shape[0])

    cpdef void flush(self):
        self.disk_array.flush()

    cpdef object next(self, object processor):
        if self.data_offset >= self.data_size:
            raise StopIteration()
        cdef uint64_t line_start = self.data_offset
        cdef uint64_t line_end = line_start
        while line_end < self.data_size and self.data_map[line_end] != 10:  # newline character
            line_end += 1
        cdef uint64_t line_length = line_end - line_start + 1  # include newline
        cdef const uint8_t* ptr = &self.data_map[0]
        self.disk_array.append(ptr, line_start, line_length)
        cdef object out = processor(self.disk_array.current_idx, self.data_map, line_start, line_length)
        self.data_offset = line_end + 1  # move past newline
        return self.disk_array.current_idx, out
