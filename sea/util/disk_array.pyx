# cython: boundscheck=False
import os
import mmap
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from sea.util.memory cimport read_uint64, read_uint32
from libcpp.utility cimport pair

cdef class DiskArray:

    def __cinit__(self, path, name="data"):

        self.data_offsets = vector[uint64_t]()
        self.data_lengths = vector[uint32_t]()
        self.data_buffer = vector[uint8_t]()

        self.path = path
        os.makedirs(path, exist_ok=True)

        self.index_file_path = os.path.join(path, f"{name}.idx")
        self.data_file_path = os.path.join(path, f"{name}.dat")
        self.index_size = 0
        self.data_size = 0
        self._open_maps()

        self.entry_size = sizeof(uint64_t) + sizeof(uint32_t)
        self.current_offset = 0
        self.current_disk_offset = self.data_size
        self.current_idx = <uint64_t>(<float>self.index_size / <float>self.entry_size)
        self.current_disk_idx = self.current_idx

    cdef void _open_maps(self):
        if self.index_size > 0:
            self.index_map.flush()
            self.index_map.close()
        if self.data_size > 0:
            self.data_map.flush()
            self.data_map.close()

        if not os.path.exists(self.index_file_path):
            open(self.index_file_path, "wb").close()
        self.index_file_read = open(self.index_file_path, "rb")
        self.index_file_write = open(self.index_file_path, "ab")
        self.index_size = os.path.getsize(self.index_file_path)
        if self.index_size > 0:
            self.index_map = mmap.mmap(self.index_file_read.fileno(), self.index_size, access=mmap.ACCESS_READ)

        if not os.path.exists(self.data_file_path):
            open(self.data_file_path, "wb").close()
        self.data_file_read = open(self.data_file_path, "rb")
        self.data_file_write = open(self.data_file_path, "ab")
        self.data_size = os.path.getsize(self.data_file_path)
        if self.data_size > 0:
            self.data_map = mmap.mmap(self.data_file_read.fileno(), self.data_size, access=mmap.ACCESS_READ)
    
    cpdef uint64_t py_append(self, bytes data):
        cdef uint64_t offset = 0
        cdef uint64_t length = len(data)
        cdef const uint8_t* ptr = <const uint8_t*>data
        return self.append(ptr, offset, length)

    cdef uint64_t append(self, const uint8_t* data, uint64_t offset, uint64_t length) noexcept nogil:
        cdef uint64_t i = offset
        while i < offset + length:
            self.data_buffer.push_back(data[i])
            i += 1
        self.data_offsets.push_back(self.current_offset)
        self.data_lengths.push_back(length)
        self.current_offset += length
        self.current_idx += 1
        return self.current_idx - 1

    cpdef uint64_t size(self):
        return self.current_idx

    cpdef bytes py_get(self, uint64_t idx):
        cdef pair[const uint8_t*, uint32_t] pair
        cdef const uint8_t* data
        cdef uint32_t length

        pair = self.get(idx)
        data = pair.first
        length = pair.second
        if length == 0:
            raise IndexError("Index out of bounds")
        return bytes(data[:length])

    cdef pair[const uint8_t*, uint32_t] get(self, uint64_t idx) noexcept nogil:
        cdef uint64_t offset
        cdef uint32_t length
        cdef uint64_t read_idx
        cdef uint64_t cur
        cdef pair[const uint8_t*, uint32_t] result # type: ignore
        cdef const uint8_t* index_ptr 

        if idx < 0 or idx >= self.current_idx:
            result.first = &self.data_buffer[0]
            result.second = 0
            return result

        if idx < self.current_disk_idx:
            cur = idx * self.entry_size
            index_ptr = &self.index_map[0]
            offset = read_uint64(index_ptr, cur)
            cur += sizeof(uint64_t)
            length = read_uint32(index_ptr, cur)
            result.first = &self.data_map[offset]
            result.second = length
        else:
            offset = self.data_offsets[idx - self.current_disk_idx]
            length = self.data_lengths[idx - self.current_disk_idx]
            result.first = &self.data_buffer[offset]
            result.second = length
        return result

    cpdef void flush(self):
        cdef size_t index_entry_size = sizeof(uint64_t) + sizeof(uint32_t)
        cdef int num_new_entries = self.current_idx - self.current_disk_idx
        cdef uint64_t idx
        cdef uint64_t offset
        cdef uint32_t length

        for i in range(num_new_entries):

            idx = self.current_disk_idx + i
            offset = self.current_disk_offset + self.data_offsets[i]
            length = self.data_lengths[i]

            self.index_file_write.write(offset.to_bytes(sizeof(uint64_t), 'little'))
            self.index_file_write.write(length.to_bytes(sizeof(uint32_t), 'little'))

        self.index_file_write.flush()

        cdef uint32_t bytes_written = 0
        for i in range(self.data_buffer.size()):
            self.data_file_write.write(self.data_buffer[i].to_bytes(1, 'little'))
            bytes_written += 1
        self.data_file_write.flush()

        self.data_buffer.clear()
        self.data_lengths.clear()
        self.data_offsets.clear()
        self.current_offset = 0
        self.current_disk_offset += bytes_written
        self.current_disk_idx = self.current_idx
        self._open_maps()

