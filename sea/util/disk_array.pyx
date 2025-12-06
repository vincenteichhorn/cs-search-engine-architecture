# cython: language_level=3
# cython: boundscheck=False
import os
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t, uint32_t, uint64_t, UINT64_MAX
from sea.util.memory cimport read_uint64, read_uint32
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy
from libc.stdio cimport fdopen, FILE, fclose, fseek, fread, SEEK_SET
from libcpp cimport bool as cbool

cdef extern from "unistd.h":
    int write(int fd, const void *buf, size_t count) nogil
    int close(int fd) nogil
    int fsync(int fd) nogil

cdef extern from "fcntl.h":
    int open(const char *pathname, int flags, int mode) nogil
    int O_WRONLY
    int O_RDWR
    int O_CREAT
    int O_RDONLY

cdef extern from "sys/mman.h": 
    int mmap(void *addr, size_t length, int prot, int flags, int fd, size_t offset) nogil 
    int munmap(void *addr, size_t length) nogil 
    int PROT_READ 
    int MAP_PRIVATE 

cdef int copen(const char *pathname, int flags, int mode) nogil:
    return open(pathname, flags, mode)

cdef int cwrite(int fd, const void *buf, size_t count) nogil:
    return write(fd, buf, count)

cdef int cclose(int fd) nogil:
    return close(fd)

cdef int cfsync(int fd) nogil:
    return fsync(fd)

cdef class DiskArray:

    def __cinit__(self, str path, str name="data", cbool open_read_maps=False):

        self.data_offsets = vector[uint64_t]()
        self.data_lengths = vector[uint32_t]()
        self.data_buffer = <uint8_t*>NULL
        self.data_buffer_size = 0
        self.data_buffer_capacity = 0

        self.path = path
        os.makedirs(path, exist_ok=True)

        self.index_file_path = os.path.join(path, f"{name}.idx")
        self.index_file_path_bytes = self.index_file_path.encode('utf-8')
        self.index_file_path_c = self.index_file_path_bytes
        self.data_file_path = os.path.join(path, f"{name}.dat")
        self.data_file_path_bytes = self.data_file_path.encode('utf-8')
        self.data_file_path_c = self.data_file_path_bytes

        self.index_fd = copen(self.index_file_path_c, O_RDWR | O_CREAT, 0o644)
        if self.index_fd == -1:
            raise OSError("Failed to open index file")
        self.index_file_ptr = fdopen(self.index_fd, "r+b")
        if self.index_file_ptr == NULL:
            raise OSError("Faile to open index file ptr")
        self.index_size = os.path.getsize(self.index_file_path)

        self.data_fd = copen(self.data_file_path_c, O_RDWR | O_CREAT, 0o644)
        if self.data_fd == -1:
            raise OSError("Failed to open index file")
        self.data_file_ptr = fdopen(self.data_fd, "r+b")
        if self.data_file_ptr == NULL:
            raise OSError("Faile to open index file ptr")
        self.data_size = os.path.getsize(self.data_file_path)

        self.open_read_maps = open_read_maps
        self.index_read_mmap = NULL
        self.data_read_mmap = NULL
        if open_read_maps:
            self._open_read_maps()

        
        self.chunk_size = 100 * 1024
        self.data_read_buffer = <BytePtr>malloc(self.chunk_size)

        self.entry_size = sizeof(uint32_t) + sizeof(uint64_t) * 2
        self.index_read_buffer = <BytePtr>malloc(self.entry_size)
        self.current_disk_offset = self.data_size

        self.current_idx = <uint64_t>(<float>self.index_size / <float>self.entry_size)
        self.current_disk_idx = self.current_idx
    
    
    def __dealloc__(self):
        if self.current_idx > self.current_disk_idx:
            self._flush()

        with nogil:
            if self.index_read_buffer != NULL:
                free(<void*>self.index_read_buffer)
            if self.data_read_buffer != NULL:
                free(<void*>self.data_read_buffer)
            if self.data_buffer != <uint8_t*>NULL:
                free(<void*>self.data_buffer)

            cclose(self.index_fd)
            cclose(self.data_fd)

            if self.index_file_ptr != NULL:
                fclose(self.index_file_ptr)
            if self.data_file_ptr != NULL:
                fclose(self.data_file_ptr)

            if self.index_read_mmap != NULL:
                munmap(<void*>self.index_read_mmap, self.index_size)
            if self.data_read_mmap != NULL:
                munmap(<void*>self.data_read_mmap, self.data_size)
    
    cdef void _open_read_maps(self) noexcept nogil:

        if self.index_read_mmap != NULL:
            munmap(<void*>self.index_read_mmap, self.index_size)
        if self.data_read_mmap != NULL:
            munmap(<void*>self.data_read_mmap, self.data_size)

        if self.index_size > 0:
            self.index_read_mmap = <BytePtr>mmap(NULL, self.index_size, PROT_READ, MAP_PRIVATE, self.index_fd, 0)
        if self.data_size > 0:
            self.data_read_mmap = <BytePtr>mmap(NULL, self.data_size, PROT_READ, MAP_PRIVATE, self.data_fd, 0)
    
    cpdef uint64_t py_append(self, uint64_t payload, bytes data):
        cdef uint64_t offset = 0
        cdef uint32_t length = len(data)
        cdef const uint8_t* ptr = <const uint8_t*>data
        return self.append(payload, ptr, offset, length)

    cdef uint64_t append(self, uint64_t payload, const uint8_t* data, uint64_t offset, uint32_t length) noexcept nogil:
        cdef uint64_t i = offset
        cdef size_t new_size = self.data_buffer_size + length
        cdef size_t new_capacity
        if new_size > self.data_buffer_capacity:
            new_capacity = max(new_size, self.data_buffer_capacity * 2 if self.data_buffer_capacity > 0 else 1024)
            self.data_buffer = <uint8_t*>realloc(self.data_buffer, new_capacity * sizeof(uint8_t))
            self.data_buffer_capacity = new_capacity
        memcpy(self.data_buffer + self.data_buffer_size, data + offset, length)

        self.payloads.push_back(payload)
        self.data_offsets.push_back(self.data_buffer_size)
        self.data_lengths.push_back(length)
        self.data_buffer_size = new_size
        self.current_idx += 1
        return self.current_idx - 1
    
    cpdef uint64_t py_add_to_last(self, bytes data):
        cdef uint64_t offset = 0
        cdef uint32_t length = len(data)
        cdef const uint8_t* ptr = <const uint8_t*>data
        return self.add_to_last(ptr, offset, length)

    cdef uint64_t add_to_last(self, const uint8_t* data, uint64_t offset, uint32_t length) noexcept nogil:

        if self.current_idx == self.current_disk_idx:
            # no entries yet
            return self.append(0, data, offset, length)

        cdef size_t new_size = self.data_buffer_size + length
        cdef size_t new_capacity
        if new_size > self.data_buffer_capacity:
            new_capacity = max(new_size, self.data_buffer_capacity * 2 if self.data_buffer_capacity > 0 else 1024)
            self.data_buffer = <uint8_t*>realloc(self.data_buffer, new_capacity * sizeof(uint8_t))
            self.data_buffer_capacity = new_capacity
        memcpy(self.data_buffer + self.data_buffer_size, data + offset, length)

        self.data_lengths[self.current_idx - self.current_disk_idx - 1] += length
        self.data_buffer_size = new_size
        return self.current_idx - 1

    cpdef uint64_t size(self):
        return self.current_idx

    cpdef object py_get(self, uint64_t idx):
        cdef EntryInfo entry
        cdef uint32_t length
        entry = self.get(idx)
        length = entry.length
        if length == 0 and entry.data == <uint8_t*>NULL:
            raise IndexError("Index out of bounds")
        if length == 0:
            return {"payload": entry.payload, "data": b"", "length": 0}
        cdef const uint8_t[:] data = <const uint8_t[:length]>entry.data #type: ignore
        return {"payload": entry.payload, "data": bytes(data), "length": length}

    cdef EntryInfo get(self, uint64_t idx) noexcept nogil:
        cdef uint64_t payload
        cdef uint64_t offset
        cdef uint32_t length
        cdef uint64_t read_idx
        cdef uint64_t cur
        cdef EntryInfo entry # type: ignore
        cdef const uint8_t* index_ptr 
        cdef size_t n
        
        if idx < 0 or idx >= self.current_idx:
            entry.payload = UINT64_MAX
            entry.data = <uint8_t*>NULL
            entry.length = 0
            return entry
        
        if idx < self.current_disk_idx:
            if self.index_file_ptr == NULL or self.data_file_ptr == NULL:
                entry.payload = UINT64_MAX
                entry.data = <uint8_t*>NULL
                entry.length = 0
                return entry

            if self.open_read_maps and self.index_read_mmap != NULL and self.data_read_mmap != NULL:
                index_ptr = self.index_read_mmap + idx * self.entry_size
                cur = 0
                payload = read_uint64(index_ptr, cur)
                cur += sizeof(uint64_t)
                offset = read_uint64(index_ptr, cur)
                cur += sizeof(uint64_t)
                length = read_uint32(index_ptr, cur)

                entry.payload = payload
                entry.data = self.data_read_mmap + offset
                entry.length = length
                return entry

            fseek(self.index_file_ptr, idx * self.entry_size, SEEK_SET)
            n = fread(<char*>self.index_read_buffer, 1, self.entry_size, self.index_file_ptr)

            index_ptr = <BytePtr>self.index_read_buffer
            cur = 0
            payload = read_uint64(index_ptr, cur)
            cur += sizeof(uint64_t)
            offset = read_uint64(index_ptr, cur)
            cur += sizeof(uint64_t)
            length = read_uint32(index_ptr, cur)

            entry.payload = payload
            fseek(self.data_file_ptr, offset, SEEK_SET)
            if length > self.chunk_size:
                self.data_read_buffer = <BytePtr>realloc(<void*>self.data_read_buffer, length)
                self.chunk_size = length
            n = fread(<char*>self.data_read_buffer, 1, length, self.data_file_ptr)
            entry.data = <uint8_t*>self.data_read_buffer
            entry.length = length
            return entry

        payload = self.payloads[idx - self.current_disk_idx]
        offset = self.data_offsets[idx - self.current_disk_idx]
        length = self.data_lengths[idx - self.current_disk_idx]
        entry.payload = payload
        entry.data = &self.data_buffer[offset]
        entry.length = length
        return entry
    
    cpdef void flush(self):
        self._flush()

    cdef void _flush(self) noexcept nogil:
        cdef uint32_t num_new_entries = self.current_idx - self.current_disk_idx
        if num_new_entries == 0:
            return
        cdef uint32_t i = 0, length, cur
        cdef uint64_t offset, payload
        cdef uint8_t * index_buffer = <uint8_t*>malloc(num_new_entries * self.entry_size)

        while i < num_new_entries:
            offset = self.current_disk_offset + self.data_offsets[i]
            length = self.data_lengths[i]
            payload = self.payloads[i]

            cur = i * self.entry_size
            memcpy(index_buffer + cur, &payload, sizeof(uint64_t))
            cur += sizeof(uint64_t)
            memcpy(index_buffer + cur, &offset, sizeof(uint64_t))
            cur += sizeof(uint64_t)
            memcpy(index_buffer + cur, &length, sizeof(uint32_t))
            i += 1
        
        cwrite(self.index_fd, index_buffer, num_new_entries * self.entry_size)
        free(index_buffer)
        cfsync(self.index_fd)

        cwrite(self.data_fd, self.data_buffer, self.data_buffer_size)
        free(self.data_buffer)
        cfsync(self.data_fd)

        self.current_disk_offset += self.data_buffer_size
        self.current_disk_idx = self.current_idx
        self.data_buffer = <uint8_t*>NULL
        self.data_buffer_size = 0
        self.data_buffer_capacity = 0
        self.data_offsets.swap(vector[uint64_t]())
        self.data_lengths.swap(vector[uint32_t]())
        self.payloads.swap(vector[uint64_t]())

        if self.open_read_maps:
            self._open_read_maps()  
    
    cdef void append_large_flush(self, uint64_t payload, const uint8_t* data, uint64_t offset, uint32_t length) noexcept nogil:
        self._flush()

        cdef uint32_t cur = 0
        cdef uint8_t * index_buffer = <uint8_t*>malloc(self.entry_size)
        cdef uint64_t index_offset = self.current_disk_offset

        memcpy(index_buffer + cur, &payload, sizeof(uint64_t))
        cur += sizeof(uint64_t)
        memcpy(index_buffer + cur, &index_offset, sizeof(uint64_t))
        cur += sizeof(uint64_t)
        memcpy(index_buffer + cur, &length, sizeof(uint32_t))
        
        cwrite(self.index_fd, index_buffer, self.entry_size)
        free(index_buffer)
        cfsync(self.index_fd)

        cwrite(self.data_fd, data, length)
        cfsync(self.data_fd)

        self.current_disk_offset += length
        self.current_disk_idx += 1
        self.current_idx += 1

        if self.open_read_maps:
            self._open_read_maps()
    
    cpdef DiskArrayIterator iterator(self):
        cdef DiskArrayIterator it = DiskArrayIterator(self)
        return it



cdef class DiskArrayIterator:

    def __cinit__(self, DiskArray disk_array):
        self.disk_array = disk_array
        self.idx = 0
        self.size = disk_array.size()
        cdef EntryInfo entry
        entry.payload = UINT64_MAX
        entry.data = <uint8_t*>NULL
        entry.length = 0
        self.current_entry = entry
    
    cdef cbool has_next(self) noexcept nogil:
        return self.idx < self.size

    cdef EntryInfo next_entry(self) noexcept nogil:

        cdef EntryInfo entry
        cdef uint8_t* tmp
        if self.idx >= self.size:
            entry.payload = UINT64_MAX
            entry.data = <uint8_t*>NULL
            entry.length = 0
            return entry
        self.current_entry = self.disk_array.get(self.idx)
        self.idx += 1
        return self.current_entry
    
    cdef EntryInfo current(self) noexcept nogil:
        cdef EntryInfo entry
        if self.idx == 0 or self.idx > self.size:
            entry.payload = UINT64_MAX
            entry.data = <uint8_t*>NULL
            entry.length = 0
            return entry
        return self.current_entry