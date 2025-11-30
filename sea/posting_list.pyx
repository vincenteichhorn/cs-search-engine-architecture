cpdef object posting_list_from_list(list items, object key=None, bint sorted=False):
    """
    Creates a PostingList from a given list of items.
    Arguments:
        items (list): A list of items to initialize the PostingList with.
        key (callable, optional): A function to extract a comparison key from each item. Defaults to None.
    Returns:
        PostingList: A new PostingList instance containing the provided items.
    """
    cdef object new_list = PostingList(key)
    new_list.items = items
    if sorted:
        new_list._sorted = True
    return new_list


cdef class PostingList:

    cdef public list items
    cdef public bint _sorted    
    cdef object _key

    
    def __cinit__(self, key=None):
        self.items = []
        self._key = key
        self._sorted = False

    cpdef void add(self, object item):
        self.items.append(item)
        self._sorted = False

    cdef ensure_sorted(self):
        """
        Ensures that the posting list is sorted based on the key function.
        """
        if not self._sorted:
            if self._key is None:
                self.items.sort()
            else:
                self.items.sort(key=self._key)
            self._sorted = True
        
    cpdef PostingList intersection(self, PostingList other, object additional_constraint = None, object merge_items = None):
        """
        Returns a new PostingList that is the intersection of this list and another.

        Arguments:
            other (PostingList): Another PostingList to intersect with.
        
        Returns:
            PostingList: A new PostingList containing items present in both lists.
        """
        self.ensure_sorted()
        other.ensure_sorted()
        cdef list new_items = []
        cdef int i = 0
        cdef int j = 0
        cdef int n = len(self.items)
        cdef int m = len(other.items)
        cdef object item1, item2, key1, key2

        while i < n and j < m:
            item1 = self.items[i]
            item2 = other.items[j]
            key1 = item1 if self._key is None else self._key(item1)
            key2 = item2 if other._key is None else other._key(item2)

            if key1 < key2:
                i += 1
            elif key1 > key2:
                j += 1
            else:
                if additional_constraint is None or additional_constraint(item1, item2):
                    new_items.append(item2 if merge_items is None else merge_items(item1, item2))
                i += 1
                j += 1

        self.items = new_items
        return self

    cpdef PostingList union(self, PostingList other, object merge_items = None):
        """
        Returns a new PostingList that is the union of this list and another.

        Arguments:
            other (PostingList): Another PostingList to unite with.
        
        Returns:
            PostingList: A new PostingList containing all unique items from both lists.
        """
        self.ensure_sorted()
        other.ensure_sorted()
        cdef list new_items = []
        cdef int i = 0
        cdef int j = 0
        cdef int n = len(self.items)
        cdef int m = len(other.items)
        cdef object item1, item2, key1, key2

        while i < n and j < m:
            item1 = self.items[i]
            item2 = other.items[j]
            key1 = item1 if self._key is None else self._key(item1)
            key2 = item2 if other._key is None else other._key(item2)

            if key1 < key2:
                new_items.append(item1)
                i += 1
            elif key1 > key2:
                new_items.append(item2)
                j += 1
            else:
                new_items.append(item2 if merge_items is None else merge_items(item1, item2))
                i += 1
                j += 1

        while i < n:
            new_items.append(self.items[i])
            i += 1

        while j < m:
            new_items.append(other.items[j])
            j += 1

        self.items = new_items
        return self

    cpdef PostingList difference(self, PostingList other):
        """
        Returns a new PostingList that is the difference of this list and another.
        Arguments:
            other (PostingList): Another PostingList to subtract from this list.
        Returns:
            PostingList: A new PostingList containing items present in this list but not in the other.
        """
        self.ensure_sorted()
        other.ensure_sorted()
        cdef list new_items = []
        cdef int i = 0
        cdef int j = 0
        cdef int n = len(self.items)
        cdef int m = len(other.items)
        cdef object item1, item2, key1, key2
        while i < n and j < m:
            item1 = self.items[i]
            item2 = other.items[j]
            key1 = item1 if self._key is None else self._key(item1)
            key2 = item2 if other._key is None else other._key(item2)

            if key1 == key2:
                i += 1
                j += 1
            elif key1 < key2:
                new_items.append(item1)
                i += 1
            else:
                j += 1

        while i < n:
            new_items.append(self.items[i])
            i += 1
                
        self.items = new_items
        return self


    cpdef clone(self):
        """
        Creates a shallow copy of the PostingList.

        Returns:
            PostingList: A new PostingList instance with the same items and key function.
        """
        self.ensure_sorted()
        cdef PostingList new_list = PostingList(self._key)
        new_list.items = self.items.copy()
        new_list._sorted = self._sorted
        return new_list

    @classmethod
    def from_list(cls, list items, object key=None, sorted=False):
        """
        Creates a PostingList from a given list of items.
        Arguments:
            items (list): A list of items to initialize the PostingList with.
            key (callable, optional): A function to extract a comparison key from each item. Defaults to None.
        Returns:
            PostingList: A new PostingList instance containing the provided items.
        """
        return posting_list_from_list(items, key, sorted)

    def __len__(self):
        self.ensure_sorted()
        return len(self.items)

    def __iter__(self):
        self.ensure_sorted()
        return iter(self.items)

    def __repr__(self) -> str:
        self.ensure_sorted()
        return f"PostingList({self.items})"
    
    def __getitem__(self, int index):
        self.ensure_sorted()
        return self.items[index]