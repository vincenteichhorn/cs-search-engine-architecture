# cython: boundscheck=False
from sea.util.disk_array cimport DiskArray
from sea.document cimport Document, TokenizedDocument, Posting
from sea.tokenizer cimport Tokenizer, TokenizedField
from libc.stdint cimport uint8_t, uint64_t, uint32_t, int64_t
from cpython.unicode cimport PyUnicode_DecodeUTF8
import os       
import mmap
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC
from libc.stdlib cimport malloc, calloc, free
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string as cstring

cpdef str py_string_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length):
    cdef const uint8_t* ptr = &data[0]
    return str(string_processor(id, ptr, offset, length), "utf-8")

cdef cstring string_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length):
    return cstring(<const char*>data + offset, length)

cpdef object py_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length):
    cdef const uint8_t* ptr = &data[0]
    cdef Document doc = document_processor(id, ptr, offset, length)
    return {
        "id": doc.id,
        "url": PyUnicode_DecodeUTF8(doc.url, doc.url_length, ""),
        "title": PyUnicode_DecodeUTF8(doc.title, doc.title_length, ""),
        "body": PyUnicode_DecodeUTF8(doc.body, doc.body_length, ""),
    }

cdef Document document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length) noexcept nogil:
    cdef const uint8_t* ptr = data + offset
    cdef uint32_t start = 0
    cdef int field_index = 0
    cdef char c
    cdef uint32_t i, j, field_len
    cdef char* src
    cdef char* buf
    cdef Document doc

    doc.id = id
    doc.url = NULL
    doc.url_length = 0
    doc.title = NULL
    doc.title_length = 0
    doc.body = NULL
    doc.body_length = 0

    i = 0
    while i < length:
        c = ptr[i]
        if c == 9 or c == 10 or c == 13:  # tab, newline, carriage return
            field_len = i - start
            if field_len > 0:
                src = <char*>ptr + start
                # allocate buffer (+1 for null-terminator)
                buf = <char*>malloc(field_len + 1)
                for j in range(field_len):
                    c = src[j]
                    buf[j] = c | (0x20 if 'A' <= c <= 'Z' else 0)  # lowercase
                buf[field_len] = 0  # null-terminate

                if field_index == 1:
                    doc.url = buf
                    doc.url_length = field_len
                elif field_index == 2:
                    doc.title = buf
                    doc.title_length = field_len
                elif field_index == 3:
                    doc.body = buf
                    doc.body_length = field_len
                else:
                    free(buf)  # extra fields we ignore
            start = i + 1
            field_index += 1
        i += 1

    return doc

cpdef object py_tokenized_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, Tokenizer tokenizer):
    cdef const uint8_t* ptr = &data[0]
    cdef TokenizedDocument tdoc = tokenized_document_processor(id, ptr, offset, length, tokenizer)
    return {
        "id": tdoc.id,
        "num_fields": tdoc.num_fields,
        "field_lengths": [tdoc.field_lengths[i] for i in range(tdoc.num_fields)],
        "tokens": [tdoc.tokens[i] for i in range(tdoc.tokens.size())],
        "postings": [
            {
                "doc_id_diff": tdoc.postings[i].doc_id_diff,
                "field_frequencies": [tdoc.postings[i].field_frequencies[j] for j in range(tdoc.num_fields)],
                "char_positions": [tdoc.postings[i].char_positions[j] for j in range(tdoc.postings[i].char_positions.size())],
                # "token_positions": [tdoc.postings[i].token_positions[j] for j in range(tdoc.postings[i].token_positions.size())],
            }
            for i in range(tdoc.postings.size())
        ],
    }

cdef TokenizedDocument tokenized_document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length, Tokenizer tokenizer) noexcept nogil:
    cdef Document doc = document_processor(id, data, offset, length)
    if doc.url_length > 0:
        free(doc.url)

    cdef TokenizedField title_field 
    if doc.title_length > 0:
        title_field = tokenizer.tokenize(<const char*>doc.title, doc.title_length, False)
        free(doc.title)
    else:
        title_field.length = 0
        title_field.max_token_id = 0

    cdef TokenizedField body_field 
    if doc.body_length > 0:
        body_field = tokenizer.tokenize(<const char*>doc.body, doc.body_length, False)
        free(doc.body)
    else:
        body_field.length = 0
        body_field.max_token_id = 0

    cdef uint32_t max_token_id = title_field.max_token_id
    if body_field.max_token_id > max_token_id:
        max_token_id = body_field.max_token_id
    cdef uint32_t num_title_tokens = title_field.length
    cdef uint32_t num_body_tokens = body_field.length

    cdef unordered_map[uint32_t, uint32_t] token_to_idx = unordered_map[uint32_t, uint32_t]()
    cdef unordered_map[uint32_t, uint32_t].iterator iter

    cdef uint32_t num_unique_tokens = 0
    cdef uint32_t i
    cdef uint32_t token
    cdef uint32_t idx
    cdef uint32_t field_idx
    cdef uint32_t char_pos 

    cdef TokenizedDocument tdoc
    tdoc.id = doc.id
    tdoc.num_fields = 2
    tdoc.field_lengths = <uint32_t*> calloc(2, sizeof(uint32_t))
    tdoc.field_lengths[0] = title_field.length
    tdoc.field_lengths[1] = body_field.length
    tdoc.tokens.reserve(num_title_tokens + num_body_tokens)
    tdoc.postings.reserve(num_title_tokens + num_body_tokens)

    cdef Posting pst
    pst.doc_id_diff = 0
    pst.field_frequencies = NULL

    i = 0
    while i < num_title_tokens + num_body_tokens:
        if i < num_title_tokens:
            field_idx = 0
            token = title_field.tokens[i]
            char_pos = title_field.char_positions[i]
        else:
            field_idx = 1
            token = body_field.tokens[i - num_title_tokens]
            char_pos = body_field.char_positions[i - num_title_tokens]
        iter = token_to_idx.find(token)
        if iter == token_to_idx.end():
            idx = num_unique_tokens
            tdoc.tokens.push_back(token)
            tdoc.postings.emplace_back(pst)
            tdoc.postings[idx].char_positions.push_back(char_pos)
            # tdoc.postings[idx].token_positions.push_back(i)
            tdoc.postings[idx].field_frequencies = <uint32_t*> calloc(2, sizeof(uint32_t))
            tdoc.postings[idx].field_frequencies[field_idx] = 1
            token_to_idx[token] = idx
            num_unique_tokens += 1
        else:
            idx = token_to_idx[token]
            tdoc.postings[idx].char_positions.push_back(char_pos)
            # tdoc.postings[idx].token_positions.push_back(i)
            tdoc.postings[idx].field_frequencies[field_idx] += 1
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

    cpdef object py_get(self, uint64_t idx, object processor):
        cdef pair[const uint8_t*, uint32_t] slice = self.disk_array.get(idx)
        cdef const uint8_t[:] data = <const uint8_t[:slice.second]>slice.first #type: ignore
        return processor(idx, data, 0, slice.second)
    
    cdef Document get_document(self, uint64_t idx) noexcept nogil:
        cdef pair[const uint8_t*, uint32_t] slice = self.disk_array.get(idx)
        return document_processor(idx, slice.first, 0, slice.second)

    cpdef void flush(self):
        self._flush()

    cdef void _flush(self) noexcept nogil:
        self.disk_array._flush()

    cpdef object py_next(self, object processor):
        cdef pair[uint64_t, uint64_t] line_info
        cdef uint64_t line_start, line_length
        line_info = self._next_line()
        line_start = line_info.first
        line_length = line_info.second

        if line_length == 0:
            raise StopIteration()

        cdef object result = processor(self.disk_array.current_idx - 1, self.data_map, line_start, line_length)

        return self.disk_array.current_idx - 1, result
    
    cdef pair[uint64_t, uint64_t] _next_line(self) noexcept nogil:
        if self.data_offset >= self.data_size:
            return pair[uint64_t, uint64_t](0, 0)

        cdef const uint8_t* ptr = &self.data_map[self.data_offset]
        cdef const uint8_t* end_ptr = &self.data_map[self.data_size]
        cdef const uint8_t* line_ptr = ptr

        while ptr < end_ptr and ptr[0] != 10: # newline
            ptr += 1

        cdef uint64_t line_start = self.data_offset
        cdef uint64_t line_end = ptr - &self.data_map[0]
        cdef uint64_t line_length = line_end - line_start + 1  # include newline

        self.data_offset = line_end + 1
        self.disk_array.append(&self.data_map[0], line_start, line_length)

        return pair[uint64_t, uint64_t](line_start, line_length)
    
    cdef pair[int64_t, TokenizedDocument] next_tokenized_document(self, Tokenizer tokenizer) noexcept nogil:
        cdef pair[uint64_t, uint64_t] line_info
        cdef uint64_t line_start, line_length
        line_info = self._next_line()
        line_start = line_info.first
        line_length = line_info.second

        cdef TokenizedDocument out
        if line_length == 0:
            out.id = -1
            out.num_fields = 0
            out.field_lengths = NULL
            return pair[int64_t, TokenizedDocument](-1, out)

        out = tokenized_document_processor(self.disk_array.current_idx - 1, &self.data_map[0], line_start, line_length, tokenizer)
        return pair[int64_t, TokenizedDocument](self.disk_array.current_idx - 1, out)
    