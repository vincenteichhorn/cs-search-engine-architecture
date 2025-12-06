from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fopen, fscanf, fclose
import mmap

cdef uint64_t read_uint64(const uint8_t* buf, Py_ssize_t offset) noexcept nogil:
    cdef const uint64_t* p = <const uint64_t*>(&buf[offset])
    return p[0]

cdef uint32_t read_uint32(const uint8_t* buf, Py_ssize_t offset) noexcept nogil:
    cdef const uint32_t* p = <const uint32_t*>(&buf[offset])
    return p[0]


cdef extern from "unistd.h":
    long sysconf(int name) nogil
    int _SC_PAGESIZE

cdef size_t get_memory_usage() noexcept nogil:
    cdef FILE* f = fopen(b"/proc/self/statm", b"r")
    if not f:
        return 0

    cdef unsigned long size, resident, shared, text, lib, data, dt
    cdef int n
    n = fscanf(f, b"%lu %lu %lu %lu %lu %lu %lu", &size, &resident, &shared, &text, &lib, &data, &dt)
    fclose(f)

    cdef long page_size = sysconf(_SC_PAGESIZE)
    if page_size <= 0:
        with gil:
            print("Warning: sysconf(_SC_PAGESIZE) failed, using fallback page size.")
        page_size = 4096  # fallback, though usually sysconf works

    return resident * page_size
