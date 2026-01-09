from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from sea.document cimport Document, TokenizedDocument, free_posting
from sea.corpus cimport Corpus, py_tokenized_document_processor
from sea.tokenizer cimport Tokenizer, TokenizedField
import numpy as np
from tqdm import tqdm
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libc.math cimport log  
from libc.stdlib cimport free
import pandas as pd

cdef get_features(vector[uint64_t] query_tokens, vector[TokenizedDocument] documents, unordered_map[uint64_t, uint64_t]& doc_freqs, uint64_t num_total_docs, vector[float]& average_field_lengths, float bm25_k, vector[float]& bm25_bs)
