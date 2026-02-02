import os
from sea.tokenizer cimport Tokenizer
from sea.corpus cimport Corpus
from sea.document cimport free_tokenized_document, TokenizedDocument, Posting, serialize_postings, BytePtr, compare_postings_ptr, update_posting_score, free_posting
from libc.stdint cimport uint64_t, uint32_t, UINT64_MAX, uint8_t
from tqdm import tqdm
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libc.stdlib cimport free, realloc
from sea.util.disk_array cimport DiskArray, DiskArrayIterator, EntryInfo
from libcpp.string cimport string as cstring
from cython.operator cimport preincrement, dereference
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort
from sea.util.memory cimport get_memory_usage
from libc.math cimport log  
import json

cdef str PARTITION_PREFIX = "partition_"
cdef str TIER_PREFIX = "tier_"
cdef list BM25_FIELD_BOOSTS = [1.0, 0.5]
cdef list BM25_BS = [0.75, 0.75]
cdef float BM25_K = 1.5

cdef int NUM_TIERS = 6
cdef list TIER_THRESHOLDS = [25.0, 20.0, 15.0, 10.0, 5.0, 0.0]

ctypedef Posting* PostingPtr

cdef extern from "malloc.h" nogil:
    int malloc_trim(size_t pad)

cdef extern from * namespace "" nogil:
    """
    #include <queue>
    #include <functional>
    #include <utility>

    using mqueue =
        std::priority_queue<
            std::pair<uint64_t, uint32_t>,
            std::vector<std::pair<uint64_t, uint32_t>>,
            std::greater<std::pair<uint64_t, uint32_t>>
        >;
    """
    cdef cppclass mqueue:
        mqueue() noexcept nogil
        void push(pair[uint64_t, uint32_t]&) nogil
        pair[uint64_t, uint32_t] top() nogil
        void pop() nogil
        bint empty() nogil
        size_t size() nogil

cdef class Indexer:
    
    cdef str save_path
    cdef uint64_t max_documents
    cdef uint32_t partition_size
    cdef uint32_t next_partition_id

    cdef Tokenizer tokenizer
    cdef Corpus corpus

    cdef object index_pbar
    cdef object merge_pbar

    cdef vector[BytePtr] free_me_later_pls
    cdef vector[Posting] postings_buffer
    cdef unordered_map[uint32_t, vector[uint32_t]] inverted_index
    cdef unordered_map[uint32_t, uint32_t] document_frequencies
    cdef uint64_t* summed_field_lengths
    cdef uint64_t num_documents
    cdef uint64_t num_postings
    cdef uint32_t num_fields

    cdef size_t current_memory_usage

    cdef DiskArray tmp_disk_array

    cdef vector[float] bm25_field_boosts
    cdef vector[float] bm25_bs
    cdef float bm25_k

    cdef uint32_t num_tiers
    cdef vector[float] tier_thresholds

    def __cinit__(self, str save_path, str dataset, uint32_t max_documents=50_000, uint32_t partition_size=10_000):
        self.save_path = save_path
        self.max_documents = max_documents
        self.partition_size = partition_size
        self.next_partition_id = 0

        os.makedirs(self.save_path, exist_ok=True)

        self.tokenizer = Tokenizer(save_path)
        self.corpus = Corpus(save_path, dataset)

        
        self.postings_buffer = vector[Posting]()
        self.free_me_later_pls = vector[BytePtr]()
        self.inverted_index = unordered_map[uint32_t, vector[uint32_t]]()
        self.document_frequencies = unordered_map[uint32_t, uint32_t]()
        self.summed_field_lengths = <uint64_t*>NULL
        self.num_fields = 0

        self.num_documents = 0
        self.num_postings = 0
        self.bm25_field_boosts = vector[float]()
        for boost in BM25_FIELD_BOOSTS:
            self.bm25_field_boosts.push_back(boost)
        self.bm25_bs = vector[float]()
        for b in BM25_BS:
            self.bm25_bs.push_back(b)
        self.bm25_k = BM25_K

        self.num_tiers = NUM_TIERS
        self.tier_thresholds = vector[float]()
        for threshold in TIER_THRESHOLDS:
            self.tier_thresholds.push_back(threshold)

    cdef void _save_metadata(self):
        cdef str metadata_path = os.path.join(self.save_path, "meta.json")
        cdef dict metadata = {}
        metadata["num_documents"] = self.num_documents
        metadata["num_postings"] = self.num_postings
        metadata["num_fields"] = self.num_fields
        metadata["bm25_k"] = self.bm25_k
        metadata["bm25_field_boosts"] = [self.bm25_field_boosts[i] for i in range(self.bm25_field_boosts.size())]
        metadata["bm25_bs"] = [self.bm25_bs[i] for i in range(self.bm25_bs.size())]
        metadata["num_tiers"] = self.num_tiers
        metadata["tier_thresholds"] = [self.tier_thresholds[i] for i in range(self.tier_thresholds.size())]
        metadata["avg_field_lengths"] = {}
        cdef uint32_t i
        for i in range(self.num_fields):
            metadata["avg_field_lengths"][str(i)] = self.summed_field_lengths[i] / self.num_documents
        metadata["term_document_frequencies"] = {}
        cdef unordered_map[uint32_t, uint32_t].iterator it = self.document_frequencies.begin()
        cdef str token
        while it != self.document_frequencies.end():
            token = self.tokenizer.py_get(dereference(it).first)
            metadata["term_document_frequencies"][token] = dereference(it).second
            preincrement(it)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
    cpdef void build(self):
        self._build()
        self._finalize()
        self._save_metadata()


    cdef void _build(self) noexcept nogil:

        with gil:
            self.index_pbar = tqdm(total=self.max_documents, desc="Indexing", unit="doc")
        
        cdef uint32_t i = 0, j = 0, k = 0
        cdef TokenizedDocument tokenized_document
        cdef uint32_t field_freq, field_len, token
        cdef clock_t start_time = clock()

        cdef size_t estimated_other_mem = 0
    
        for i in range(self.max_documents):
            tokenized_document = self.corpus.next_tokenized_document(self.tokenizer)
            if tokenized_document.id == UINT64_MAX:
                free_tokenized_document(&tokenized_document)
                break
            for j in range(tokenized_document.tokens.size()):
                self.postings_buffer.push_back(tokenized_document.postings[j])
                token = tokenized_document.tokens[j]
                if self.inverted_index.find(token) == self.inverted_index.end():
                    self.inverted_index[token] = vector[uint32_t]()
                    self.document_frequencies[token] = 0
                self.document_frequencies[token] += 1
                self.inverted_index[token].push_back(self.postings_buffer.size() - 1)
                self.num_postings += 1
            self.num_documents += 1
                
            if tokenized_document.num_fields > self.num_fields:
                self.summed_field_lengths = <uint64_t*>realloc(self.summed_field_lengths, tokenized_document.num_fields * sizeof(uint64_t))
            for k in range(tokenized_document.num_fields):
                if k >= self.num_fields:
                    self.summed_field_lengths[k] = 0
                self.summed_field_lengths[k] += tokenized_document.field_lengths[k]
            self.num_fields = tokenized_document.num_fields
            self.free_me_later_pls.push_back(<BytePtr>tokenized_document.field_lengths)
            
            with gil:
                self.index_pbar.update(1)
                        
            if i % self.partition_size == 0 and i > 0:
                self._flush()

            if i % 1000 == 0:
                self.current_memory_usage = get_memory_usage()
                with gil:
                    self.index_pbar.set_postfix({"Mem (MB)": f"{self.current_memory_usage / (1024*1024):.2f}"})

        cdef clock_t end_time = clock()
        cdef double total_time = (end_time - start_time) / CLOCKS_PER_SEC
    
        with gil:
            self.index_pbar.close()

        if self.inverted_index.size() > 0:
            self._flush()
        
        # with gil:
        #     print(f"Avg. Docs/sec: {i / total_time:.2f}")
        #     print(f"Estimated indexing time for 3M documents: {(total_time/i*3_000_000/60) if i>0 else 0:.2f} minutes")
        
        
    cdef void _flush(self) noexcept nogil:

        self.corpus._flush()
        self.tokenizer._flush()

        cdef unordered_map[uint32_t, vector[uint32_t]].iterator it = self.inverted_index.begin()
        cdef vector[uint32_t] tokens = vector[uint32_t]()
        tokens.reserve(self.inverted_index.size())
        while it != self.inverted_index.end():
            tokens.push_back(dereference(it).first)
            preincrement(it)

        sort(tokens.begin(), tokens.end())

        cdef uint32_t token
        cdef vector[PostingPtr] posting_ptrs
        cdef vector[uint32_t] posting_indices
        cdef cstring partition_name
        cdef uint8_t* buffer
        cdef pair[BytePtr, uint32_t] serialized_postings
        cdef uint64_t last_doc_id = 0
        with gil:
            self.tmp_disk_array = DiskArray(os.path.join(self.save_path, f"{PARTITION_PREFIX}{self.next_partition_id}"))
        self.next_partition_id += 1
        
        cdef uint32_t i, j

        for i in range(tokens.size()):

            token = tokens[i]
            it = self.inverted_index.find(token)
            posting_indices = dereference(it).second
            posting_ptrs = vector[PostingPtr]()
            posting_ptrs.reserve(posting_indices.size())
            for j in range(posting_indices.size()):
                posting_ptrs.push_back(&self.postings_buffer[posting_indices[j]])

            sort(posting_ptrs.begin(), posting_ptrs.end(), compare_postings_ptr)

            # last_doc_id = 0
            # for j in range(posting_ptrs.size()):
            #     posting_ptrs[j].doc_id -= last_doc_id
            #     last_doc_id += posting_ptrs[j].doc_id
            
            serialized_postings = serialize_postings(&posting_ptrs)
            buffer = serialized_postings.first
            self.tmp_disk_array.append(token, buffer, 0, serialized_postings.second)
            free(buffer)

            for j in range(posting_ptrs.size()):
                free_posting(posting_ptrs[j], False)
            posting_ptrs.clear()
        
        for i in range(self.free_me_later_pls.size()):
            free(<void*>self.free_me_later_pls[i])

        self.postings_buffer.swap(vector[Posting]())
        self.free_me_later_pls.swap(vector[BytePtr]())
        self.inverted_index.swap(unordered_map[uint32_t, vector[uint32_t]]())
        self.tmp_disk_array._flush()

        malloc_trim(0)

        # with gil:
        #     self.tmp_disk_array = None
        #     gc.collect()
        #     os.system(f"pmap -x {os.getpid()} > {os.path.join(self.save_path, f'heap_{PARTITION_PREFIX}{self.next_partition_id-1}.txt')}")

    cdef uint32_t _score_to_tier_idx(self, float score) noexcept nogil:
        cdef uint32_t tier_idx = self.num_tiers - 1
        cdef uint32_t i
        for i in range(self.num_tiers):
            if score >= self.tier_thresholds[i]:
                tier_idx = i
                break
        return tier_idx

    cdef void _finalize(self):
        
        cdef uint32_t i
        cdef uint32_t num_partitions = self.next_partition_id
        cdef pair[uint64_t, uint32_t] current_merge
        cdef uint32_t current_partition_id
        cdef uint64_t current_token_id = UINT64_MAX
        cdef uint64_t tmp_id
        cdef uint32_t current_token_df = 0
        cdef float current_token_idf = 0.0
        cdef pair[float, size_t] score_and_size
        cdef uint32_t num_postings_done = 0
        cdef uint32_t last_flush_postings_done = 0
        cdef uint32_t min_flush_interval = 100_000


        cdef size_t cur = 0

        cdef list partition_disk_arrays = []
        cdef list partition_disk_array_iterators = []
        cdef DiskArrayIterator tmp_iterator
        cdef EntryInfo tmp_entry

        for i in range(num_partitions):
            partition_disk_arrays.append(DiskArray(os.path.join(self.save_path, f"{PARTITION_PREFIX}{i}")))
            partition_disk_array_iterators.append(DiskArrayIterator(<DiskArray>partition_disk_arrays[i]))
        
        cdef list tier_disk_arrays = []
        cdef DiskArray tmp_disk_array
        for i in range(self.num_tiers):
            tier_disk_arrays.append(DiskArray(os.path.join(self.save_path, f"{TIER_PREFIX}{i}")))

        cdef vector[float] average_field_lengths = vector[float]()
        cdef mqueue merge_queue = mqueue() # pair of (partition_id, token_id)

        self.merge_pbar = tqdm(total=self.num_postings, desc="Merging", unit="posting")

        with nogil:
            for i in range(self.num_fields):
                average_field_lengths.push_back(self.summed_field_lengths[i] / self.num_documents)

            for i in range(num_partitions):
                with gil:
                    tmp_iterator = <DiskArrayIterator>partition_disk_array_iterators[i]
                tmp_entry = tmp_iterator.next_entry()
                current_merge = pair[uint64_t, uint32_t](tmp_entry.payload, i)
                if current_merge.first != UINT64_MAX:
                    merge_queue.push(current_merge)
                
            while not merge_queue.empty():
                current_merge = merge_queue.top()

                if current_merge.first != current_token_id:
                    current_token_df = dereference(self.document_frequencies.find(current_merge.first)).second
                    current_token_idf = log((self.num_documents - current_token_df + 0.5) / (current_token_df + 0.5))

                    for i in range(self.num_tiers):
                        with gil:
                            tmp_disk_array = <DiskArray>tier_disk_arrays[i]
                        if num_postings_done > last_flush_postings_done + min_flush_interval:
                            tmp_disk_array._flush()
                        tmp_id = tmp_disk_array.append(current_token_df, <BytePtr>NULL, 0, 0)
                        assert tmp_id == current_merge.first
                    if num_postings_done > last_flush_postings_done + min_flush_interval:
                        last_flush_postings_done = num_postings_done
                    current_token_id = current_merge.first

                merge_queue.pop()
                current_token_id = current_merge.first
                current_partition_id = current_merge.second
                with gil:
                    tmp_iterator = <DiskArrayIterator>partition_disk_array_iterators[current_partition_id]
                tmp_entry = tmp_iterator.current()

                cur = 0
                while cur < tmp_entry.length:
                    score_and_size = update_posting_score(tmp_entry.data, cur, current_token_idf, self.bm25_k, self.bm25_field_boosts, self.bm25_bs, average_field_lengths)
                    
                    with gil:
                        tmp_disk_array = <DiskArray>tier_disk_arrays[self._score_to_tier_idx(score_and_size.first)]
                    tmp_disk_array.add_to_last(tmp_entry.data, cur, score_and_size.second)
                    cur += score_and_size.second
                    with gil:
                        self.merge_pbar.update(1)
                    num_postings_done += 1

                with gil:
                    tmp_iterator = <DiskArrayIterator>partition_disk_array_iterators[current_partition_id]
                tmp_entry = tmp_iterator.next_entry()
                current_merge = pair[uint64_t, uint32_t](tmp_entry.payload, current_partition_id)
                if current_merge.first != UINT64_MAX:
                    merge_queue.push(current_merge)
                
                if num_postings_done % 1000 == 0:
                    self.current_memory_usage = get_memory_usage()
                    with gil:
                        self.merge_pbar.set_postfix({"Mem (MB)": f"{self.current_memory_usage / (1024*1024):.2f}"})
                
            for i in range(self.num_tiers):
                with gil:
                    tmp_disk_array = <DiskArray>tier_disk_arrays[i]
                tmp_disk_array._flush()
