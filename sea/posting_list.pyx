cdef class PostingList:

    cdef list _items
    cdef object _key
    
    def __cinit__(self, key=None):
        self._items = []
        self._key = key

    cpdef void add(self, object item):
        """
        Adds an item to the posting list while maintaining sorted order. Sorting is
        based on the provided key function, or the item itself if no key is given.

        Arguments:
            item (object): The item to be added to the posting list.
        """

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
        """
        Returns a new PostingList that is the intersection of this list and another.

        Arguments:
            other (PostingList): Another PostingList to intersect with.
        
        Returns:
            PostingList: A new PostingList containing items present in both lists.
        """
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

    cpdef PostingList positional_intersection(self, PostingList other, str self_token, str other_token, int k=1):
        """
        Returns a new PostingList that is the intersection of this list and another.

        Arguments:
            other (PostingList): Another PostingList to intersect with.
        
        Returns:
            PostingList: A new PostingList containing items present in both lists.
        """
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
                if self.contains_phrase(item1, self_token, other_token, k):
                    new_items.append(item1)
                i += 1
                j += 1

        self._items = new_items
        return self

    cpdef int contains_phrase(self, object item, str self_token, str other_token, int k=1):
        """
        Checks if the Document contains the phrase defined by the positions of two tokens.

        Arguments:
            item (Document): The Document to check.
            self_token (str): The first token in the phrase.
            other_token (str): The second token in the phrase.
            k (int): The positional distance between the two tokens.

        Returns:
            bool: True if the phrase exists in the PostingList, False otherwise.
        """
        for pos1 in item.token_positions[self_token]:
            for pos2 in item.token_positions[other_token]:
                if pos1 + k == pos2:
                    return 1
        return 0

    cpdef PostingList union(self, PostingList other):
        """
        Returns a new PostingList that is the union of this list and another.

        Arguments:
            other (PostingList): Another PostingList to unite with.
        
        Returns:
            PostingList: A new PostingList containing all unique items from both lists.
        """
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
                new_items.append(item1)
                i += 1
            elif key1 > key2:
                new_items.append(item2)
                j += 1
            else:
                new_items.append(item1)
                i += 1
                j += 1

        while i < n:
            new_items.append(self._items[i])
            i += 1

        while j < m:
            new_items.append(other._items[j])
            j += 1

        self._items = new_items
        return self

    cpdef PostingList difference(self, PostingList other):
        """
        Returns a new PostingList that is the difference of this list and another.
        Arguments:
            other (PostingList): Another PostingList to subtract from this list.
        Returns:
            PostingList: A new PostingList containing items present in this list but not in the other.
        """
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

            if key1 == key2:
                i += 1
                j += 1
            elif key1 < key2:
                new_items.append(item1)
                i += 1
            else:
                j += 1

        while i < n:
            new_items.append(self._items[i])
            i += 1
                
        self._items = new_items
        return self


    cpdef clone(self):
        """
        Creates a shallow copy of the PostingList.

        Returns:
            PostingList: A new PostingList instance with the same items and key function.
        """
        cdef PostingList new_list = PostingList(self._key)
        new_list._items = self._items.copy()
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
        cdef PostingList new_list = cls(key)
        if not sorted:
            for item in items:
                new_list.add(item)
            return new_list
        new_list._items = items.copy()
        return new_list

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:
        return f"PostingList({self._items})"
    
    def __getitem__(self, int index):
        return self._items[index]