# cython: boundscheck=False
from sea.util.disk_array cimport DiskArray
from sea.document cimport Document, TokenizedDocument, Posting, free_tokenized_document, free_document
from sea.tokenizer cimport Tokenizer, TokenizedField
from libc.stdint cimport uint8_t, uint64_t, uint32_t, int64_t, UINT64_MAX
from cpython.unicode cimport PyUnicode_DecodeUTF8
import os       
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC
from libc.stdlib cimport malloc, calloc, free
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string as cstring
from sea.util.disk_array cimport EntryInfo
from libc.stdio cimport fdopen, fgets, fclose, fseek
from libc.string cimport strlen
from libc.stdlib cimport atoi

cpdef str py_string_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length):
    cdef const uint8_t* ptr = &data[0]
    return str(string_processor(id, ptr, offset, length), "utf-8")

cdef cstring string_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length) noexcept nogil:
    return cstring(<const char*>data + offset, length)

cpdef object doc_to_dict(Document doc):
    cdef object result = {
        "id": doc.id,
        "score": doc.score,
        "url": PyUnicode_DecodeUTF8(doc.url, doc.url_length, ""),
        "title": PyUnicode_DecodeUTF8(doc.title, doc.title_length, ""),
        "body": PyUnicode_DecodeUTF8(doc.body, doc.body_length, ""),
        "snippet": PyUnicode_DecodeUTF8(doc.snippet, doc.snippet_length, ""),
    }
    free_document(&doc)
    return result

cpdef object py_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, bint lowercase):
    cdef const uint8_t* ptr = &data[0]
    cdef Document doc = document_processor(id, ptr, offset, length, lowercase)
    cdef object result = doc_to_dict(doc)
    return result

cdef Document document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length, bint lowercase) noexcept nogil:
    cdef const uint8_t* ptr = data + offset
    cdef uint32_t start = 0
    cdef int field_index = 0
    cdef char c
    cdef uint32_t i, j, field_len
    cdef char* src
    cdef char* buf
    cdef Document doc

    doc.id = id
    doc.score = 0.0
    doc.url = NULL
    doc.url_length = 0
    doc.title = NULL
    doc.title_length = 0
    doc.body = NULL
    doc.body_length = 0
    doc.snippet = NULL
    doc.snippet_length = 0

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
                    buf[j] = c | (0x20 if 'A' <= c <= 'Z' and lowercase else 0)  # lowercase
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
    cdef object result = {
        "id": tdoc.id,
        "num_fields": tdoc.num_fields,
        "field_lengths": [tdoc.field_lengths[i] for i in range(tdoc.num_fields)],
        "tokens": [tdoc.tokens[i] for i in range(tdoc.tokens.size())],
        "postings": [
            {
                "doc_id": tdoc.postings[i].doc_id,
                "score": tdoc.postings[i].score,
                "num_fields": tdoc.postings[i].num_fields,
                "field_frequencies": [tdoc.postings[i].field_frequencies[j] for j in range(tdoc.postings[i].num_fields)],
                "char_positions": [tdoc.postings[i].char_positions[j] for j in range(tdoc.postings[i].char_positions.size())],
                # "token_positions": [tdoc.postings[i].token_positions[j] for j in range(tdoc.postings[i].token_positions.size())],
            }
            for i in range(tdoc.postings.size())
        ],
    }
    free_tokenized_document(&tdoc)
    return result

cdef TokenizedDocument tokenized_document_processor(uint64_t id, const uint8_t* data, uint64_t offset, uint64_t length, Tokenizer tokenizer) noexcept nogil:
    cdef Document doc = document_processor(id, data, offset, length, True)
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
    pst.doc_id = doc.id
    pst.score = 0.0
    pst.num_fields = 2
    pst.field_frequencies = NULL
    pst.field_lengths = tdoc.field_lengths

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

cdef extern from "fcntl.h":
    int open(const char *pathname, int flags, int mode) nogil
    int O_RDONLY

cdef int copen(const char *pathname, int flags, int mode) nogil:
    return open(pathname, flags, mode)

cdef class Corpus:
    
    def __cinit__(self, save_path, data_file_path, mmap=False):
        self.save_path = save_path
        self.data_file_path = data_file_path
        self.disk_array = DiskArray(save_path, name="corpus", open_read_maps=mmap)

        if os.path.exists(data_file_path):
            self.data_size = os.path.getsize(data_file_path)
            self.data_file_fd = copen(data_file_path.encode('utf-8'), O_RDONLY, 0)
            if self.data_file_fd < 0:
                raise OSError("Failed to open data file: " + data_file_path)
            self.data_file_ptr = fdopen(self.data_file_fd, "r")
            if self.data_file_ptr == NULL:
                raise OSError("Failed to open data file pointer: " + data_file_path)
            self.max_line_length = 100 * 1024 * 1024  # 10 MB
            self.line_buffer = <BytePtr>malloc(self.max_line_length)
            self.data_offset = self.disk_array.data_size
            if self.data_offset > self.data_size:
                self.data_offset = self.data_size
            if self.data_offset > 0:
                fseek(self.data_file_ptr, self.data_offset, 0)
            self.serve = False
        else:
            self.data_size = 0
            self.data_file_fd = -1
            self.data_file_ptr = NULL
            self.max_line_length = 0
            self.line_buffer = NULL
            self.data_offset = 0
            self.serve = True
    
    def __dealloc__(self):
        if self.data_file_ptr != NULL:
            fclose(self.data_file_ptr)
        if self.line_buffer != NULL:
            free(self.line_buffer)

    cpdef object py_get(self, uint64_t idx, object processor):
        cdef EntryInfo slice = self.disk_array.get(idx)
        cdef const uint8_t[:] data = <const uint8_t[:slice.length]>slice.data #type: ignore
        return processor(idx, data, 0, slice.length)
    
    cpdef object py_get_document(self, uint64_t idx, bint lowercase):
        cdef EntryInfo slice = self.disk_array.get(idx)
        cdef const uint8_t[:] data = <const uint8_t[:slice.length]>slice.data #type: ignore
        return py_document_processor(idx, data, 0, slice.length, lowercase)
    
    cdef Document get_document(self, uint64_t idx, bint lowercase) noexcept nogil:
        cdef EntryInfo slice = self.disk_array.get(idx)
        return document_processor(idx, slice.data, 0, slice.length, lowercase)
    
    cdef TokenizedDocument get_tokenized_document(self, uint64_t idx, Tokenizer tokenizer) noexcept nogil:
        cdef EntryInfo slice = self.disk_array.get(idx)
        return tokenized_document_processor(idx, slice.data, 0, slice.length, tokenizer)

    cpdef void flush(self):
        self._flush()

    cdef void _flush(self) noexcept nogil:
        self.disk_array._flush()

    cpdef object py_next(self, object processor):
        if self.serve:
            raise RuntimeError("No valid data file to read from.")

        cdef pair[BytePtr, uint64_t] line_info
        line_info = self._next_line()
        cdef BytePtr data_ptr = line_info.first
        cdef uint64_t line_length = line_info.second

        if line_length == 0:
            raise StopIteration()
        cdef const uint8_t[:] data = <const uint8_t[:line_length]>data_ptr # type: ignore

        cdef object result = processor(self.disk_array.current_idx - 1, data, 0, line_length)

        return result
    
    cdef pair[BytePtr, uint64_t] _next_line(self) noexcept nogil:
        if self.data_offset >= self.data_size:
            return pair[BytePtr, uint64_t](NULL, 0)

        self.line_buffer = <BytePtr>fgets(<char*>self.line_buffer, self.max_line_length, self.data_file_ptr)
        cdef size_t str_len = strlen(<char*>self.line_buffer)

        self.disk_array.append(0, self.line_buffer, 0, str_len)
        self.data_offset += str_len

        return pair[BytePtr, uint64_t](self.line_buffer, str_len)
    
    cdef TokenizedDocument next_tokenized_document(self, Tokenizer tokenizer) noexcept nogil:

        cdef TokenizedDocument out
        if self.serve:
            out.id = UINT64_MAX
            out.num_fields = 0
            out.field_lengths = NULL
            return out

        cdef pair[BytePtr, uint64_t] line_info
        line_info = self._next_line()
        cdef BytePtr data_ptr = line_info.first
        cdef size_t line_length = line_info.second

        if line_length == 0:
            out.id = UINT64_MAX
            out.num_fields = 0
            out.field_lengths = NULL
            return out

        out = tokenized_document_processor(self.disk_array.current_idx - 1, self.line_buffer, 0, line_length, tokenizer)

        return out
    