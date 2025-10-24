

cdef class PostingList:

    cdef list _items
    cdef object _key
    
    def __cinit__(self, key=None):
        self._items = []
        self._key = key

    cpdef void add(self, object item):
        cdef object object_key = item if self._key is None else self._key(item)
        cdef int lo = 0
        cdef int hi = len(self._items)
        cdef int mid
        cdef object mid_item, mid_key

        while lo < hi:
            mid = (lo + hi) // 2
            mid_item = self._items[mid]
            mid_key = mid_item if self._key is None else self._key(mid_item)
            if object_key < mid_key:
                hi = mid
            elif object_key > mid_key:
                lo = mid + 1
            else:
                return  # Item already exists

        self._items.insert(lo, item)

    cpdef PostingList intersection(self, PostingList other):
        cdef list new_items = []
        cdef int i = 0
        cdef int j = 0
        cdef int n = len(self._items)
        cdef int m = len(other._items)
        cdef object item1, item2, key1, key2

        while i < n and j < m:
            item1 = self._items[i]
            item2 = other._items[j]
            key1 = item1 if self._key is None else self._key(item1)
            key2 = item2 if other._key is None else other._key(item2)

            if key1 < key2:
                i += 1
            elif key1 > key2:
                j += 1
            else:
                new_items.append(item1)
                i += 1
                j += 1

        self._items = new_items
        return self

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:
        return f"SortedSet({self._items})"