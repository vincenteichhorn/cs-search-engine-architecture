from sea.disk_array cimport DiskArray
from sea.util.memory cimport SmartBuffer
from sea.document cimport Document, TokenizedDocument, TokenInfo
from sea.tokenizer cimport Tokenizer, TokenizedField
from libc.stdint cimport uint8_t, uint64_t, uint32_t
from libcpp.string cimport string as cstring
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from cpython.unicode cimport PyUnicode_DecodeUTF8
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
import os
import mmap

cpdef str identity_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length):
    cdef const uint8_t* ptr = &data[offset]
    return PyUnicode_DecodeUTF8(<const char*>ptr, length, "")

cpdef Document document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length):
    cdef const uint8_t* ptr = &data[offset]
    cdef cstring line = cstring(<const char*>ptr, length)
    cdef cstring[4] parts
    cdef int i = 0
    cdef int start = 0
    cdef int end = 0
    for j in range(length):
        if line[j] == '\t':
            parts[i] = line.substr(start, j - start)
            i += 1
            start = j + 1
    parts[i] = line.substr(start, length - start - 2).strip() # exclude "\r\n"
    cdef Document doc
    doc.id = id
    doc.url = parts[1]
    doc.title = parts[2]
    doc.body = parts[3]
    return doc

cpdef TokenizedDocument tokenized_document_processor(uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length, Tokenizer tokenizer):
    
    cdef Document doc = document_processor(id, data, offset, length)

    cdef TokenizedField title_field = tokenizer.tokenize(doc.title, is_query=False)
    cdef TokenizedField body_field = tokenizer.tokenize(doc.body, is_query=False)

    cdef unordered_map[uint32_t, TokenInfo] token_map = unordered_map[uint32_t, TokenInfo]()

    cdef uint32_t num_title_tokens = title_field.length
    cdef TokenInfo info
    cdef pair[unordered_map[uint32_t, TokenInfo].iterator, bint] result
    cdef uint32_t i
    cdef uint32_t token

    for i in range(title_field.length):
        token = title_field.tokens[i]
        info.char_position = title_field.char_positions[i]
        info.token_position = i
        info.frequency = 1
        result = token_map.insert((token, info))
        if not result.second:
            dereference(result.first).second.frequency += 1
    for i in range(title_field.length):
        token = title_field.tokens[i]
        info.char_position = title_field.char_positions[i]
        info.token_position = num_title_tokens + i
        info.frequency = 1
        result = token_map.insert((token, info))
        if not result.second:
            dereference(result.first).second.frequency += 1

    cdef TokenizedDocument tdoc
    tdoc.id = doc.id
    tdoc.field_lengths.reserve(2)
    tdoc.field_lengths.push_back(title_field.length)
    tdoc.field_lengths.push_back(body_field.length)

    tdoc.tokens.reserve(token_map.size())
    tdoc.token_infos.reserve(token_map.size())

    cdef unordered_map[uint32_t, TokenInfo].iterator it = token_map.begin()
    i = 0
    while it != token_map.end():
        tdoc.tokens.push_back(dereference(it).first)
        tdoc.token_infos.push_back(dereference(it).second)
        i += 1
        preincrement(it)

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
        cdef SmartBuffer buffer = self.disk_array.get(idx)
        return processor(idx, buffer.ptr, 0, buffer.size)

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
