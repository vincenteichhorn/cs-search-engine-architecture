import os
from sea.tokenizer cimport Tokenizer
from sea.corpus cimport Corpus, document_processor, tokenized_document_processor, TokenizedDocument, Document
from sea.util.memory cimport SmartBuffer
from libc.stdint cimport uint64_t, uint32_t, uint8_t
from tqdm import tqdm
from libc.time cimport clock, clock_t, CLOCKS_PER_SEC

cdef class Indexer:
    
    cdef str save_path
    cdef uint64_t max_documents
    cdef Tokenizer tokenizer

    cdef double _proc_time
    cdef double _total_time

    def __cinit__(self, save_path, uint32_t max_documents=10_000):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.tokenizer = Tokenizer(save_path)
        self.max_documents = max_documents
        self._proc_time = 0.0
        self._total_time = 0.0

    cdef processor_wrapper(self, uint64_t id, const uint8_t[:] data, uint64_t offset, uint64_t length):
        cdef clock_t t1 = clock()
        res = tokenized_document_processor(id, data, offset, length, self.tokenizer)
        # res = document_processor(id, data, offset, length)
        cdef clock_t t2 = clock()
        self._proc_time += (t2 - t1) / CLOCKS_PER_SEC
        return res

    cpdef build(self, Corpus corpus):
       
        cdef uint64_t id = 0, i = 0
        cdef TokenizedDocument tokenized_document
        # cdef Document tokenized_document
        cdef clock_t start_time = clock()
    
        try:
            while id < self.max_documents:
                id, tokenized_document = corpus.next(self.processor_wrapper)
                # print(f"Indexed document ID: {id}, Tokens: {tokenized_document.tokens.size()}")
                # print("First 10 tokens:", end=" ")
                # for i in range(min(10, tokenized_document.tokens.size())):
                #     print(self.tokenizer.get(tokenized_document.tokens[i]), end=" ")
                # print()
        except StopIteration:
            pass
        cdef clock_t end_time = clock()
        self._total_time = (end_time - start_time) / CLOCKS_PER_SEC

        print(f"Indexing completed in {self._total_time:.2f} seconds.")
        print(f"Docs/sec: {id / self._total_time:.2f}")
        print(f"Tokenization time: {self._proc_time:.2f}s ({(self._proc_time/self._total_time*100) if self._total_time>0 else 0:.1f}%)")
        print(f"Average time per document: {(self._total_time/id) if id>0 else 0:.8f}s")
        print(f"Estimated indexing time for 3M documents: {(self._total_time/id*3_000_000/60) if id>0 else 0:.2f}minutes")
