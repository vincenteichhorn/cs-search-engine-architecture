import os
from sea.tokenizer cimport Tokenizer
from sea.corpus cimport Corpus, document_processor, tokenized_document_processor, TokenizedDocument, Document
from libc.stdint cimport uint64_t, uint32_t, uint8_t, int64_t
from tqdm import tqdm
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC
from libcpp.utility cimport pair

cdef class Indexer:
    
    cdef str save_path
    cdef uint64_t max_documents
    cdef Tokenizer tokenizer

    cdef double _total_time

    cdef object index_pbar
    cdef object merge_pbar

    def __cinit__(self, save_path, uint32_t max_documents=10_000):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.tokenizer = Tokenizer(save_path)
        self.max_documents = max_documents
        self._total_time = 0.0
        self.index_pbar = tqdm(total=self.max_documents, desc="Indexing", unit="doc")
        self.merge_pbar = tqdm(total=0, desc="Merging", unit="posting")
    
    cpdef void build(self, Corpus corpus):
        self._build(corpus)

    cdef void _build(self, Corpus corpus) noexcept nogil:
       
        cdef uint64_t id = 0, i = 0
        cdef pair[int64_t, TokenizedDocument] pair
        cdef TokenizedDocument tokenized_document
        cdef clock_t start_time = clock()
    
        while id < self.max_documents:
            pair = corpus.next_tokenized_document(self.tokenizer)
            id = pair.first
            tokenized_document = pair.second
        
            with gil:
                self.index_pbar.update(1)
            # if tokenized_document.tokens.size() > 0:
                # with gil:
                #     print(f"Tokenized Document (ID: {tokenized_document.id}):")
                #     print(f"  Number of tokens: {tokenized_document.tokens.size()}")
                #     print(f"  Field lengths: {[tokenized_document.field_lengths[j] for j in range(tokenized_document.num_fields)]}")
                #     print(f"  First token ID: {tokenized_document.tokens[0]}")
                #     print(f"  First token info: positions={tokenized_document.postings[0].char_positions}, field_frequencies={[tokenized_document.postings[0].field_frequencies[j] for j in range(tokenized_document.num_fields)]}")
                #     i += 1

        cdef clock_t end_time = clock()
        self._total_time = (end_time - start_time) / CLOCKS_PER_SEC
    
        with gil:
            self.index_pbar.close()
        
        with gil:
            print(f"Indexing completed in {self._total_time:.2f} seconds.")
            print(f"Docs/sec: {id / self._total_time:.2f}")
            print(f"Average time per document: {(self._total_time/id) if id>0 else 0:.8f}s")
            print(f"Estimated indexing time for 3M documents: {(self._total_time/id*3_000_000/60) if id>0 else 0:.2f} minutes")
