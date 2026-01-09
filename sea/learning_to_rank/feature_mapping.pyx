# cython: boundscheck=False, wraparound=False, cdivision=True
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


cpdef build_dataset(
        index_path,
        dataset_path,
        df,
        num_docs,
        bm25_k,
        bm25_bs,
        average_field_lengths,
        document_frequencies,
    ):

    print("Starting dataset build...")

    cdef Corpus corpus = Corpus(index_path, "", mmap=False)
    print("Corpus loaded.")
    cdef Tokenizer tokenizer = Tokenizer(index_path, mmap=False)
    print("Tokenizer loaded.")

    cdef unordered_map[uint64_t, uint64_t] doc_freqs = unordered_map[uint64_t, uint64_t]()
    cdef uint64_t token_id
    cdef bytes token_bytes
    cdef char* c_token
    for token_str, dfreq in document_frequencies.items():
        token_bytes = token_str.encode("utf-8")
        c_token = token_bytes
        token_id = tokenizer.vocab_lookup(c_token, len(token_bytes))
        doc_freqs[token_id] = dfreq
    
    print("Document frequencies loaded. Total unique tokens: ", doc_freqs.size())
    
    cdef bytes query_bytes
    cdef char* c_query
    cdef vector[uint64_t] c_query_tokens

    cdef vector[float] v_average_field_lengths = vector[float]()
    cdef float avg_len
    for _, avg_len in average_field_lengths.items():
        v_average_field_lengths.push_back(avg_len)
    
    cdef vector[float] v_bm25_bs = vector[float]()
    cdef float b
    for b in bm25_bs:
        v_bm25_bs.push_back(b)
    
    cdef uint32_t i, j, num_docs_local, num_features_local

    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write("bm25_title,bm25_body,title_length,body_length,query_in_title,query_id,our_id,rank\n")  # header
        for query_id, group in tqdm(
            df.groupby("query_id"),
            desc="Building dataset",
            unit=" queries",
            total=df["query_id"].nunique(),
        ):
            query_text = group["query_text"].values[0]
            doc_ids = group["our_id"].tolist()
            documents = get_documents(doc_ids, corpus, tokenizer)
            query_bytes = query_text.encode("utf-8")
            c_query = query_bytes
            c_query_tokens = tokenizer.tokenize(c_query, len(query_text), True).tokens
            feature_matrix = get_features(c_query_tokens, documents, doc_freqs, num_docs, v_average_field_lengths, bm25_k, v_bm25_bs)

            # write features to file
            num_docs_local = feature_matrix.shape[0]
            num_features_local = feature_matrix.shape[1]
            ranks = group['rank'].tolist()
            for i in range(num_docs_local):
                for j in range(num_features_local):
                    f.write(f"{feature_matrix[i, j]}")
                    if j < num_features_local - 1:
                        f.write(",")
                f.write(f",{query_id},{doc_ids[i]},{ranks[i]}\n")
            
            for doc in documents:
                for i in range(doc.postings.size()):
                    free_posting(&doc.postings[i], False)
                # free(&doc.field_lengths)
            

cdef vector[TokenizedDocument] get_documents(list doc_ids, Corpus corpus, Tokenizer tokenizer):
    cdef vector[TokenizedDocument] c_docs = vector[TokenizedDocument]()
    cdef TokenizedDocument tokenized_doc

    for doc_id in doc_ids:
        tokenized_doc = corpus.get_tokenized_document(<uint64_t>doc_id, tokenizer)
        c_docs.push_back(tokenized_doc)
    
    return c_docs

cdef get_features(vector[uint64_t] query_tokens, vector[TokenizedDocument] documents, unordered_map[uint64_t, uint64_t]& doc_freqs, uint64_t num_total_docs, vector[float]& average_field_lengths, float bm25_k, vector[float]& bm25_bs):

    # return a dummy zero matrix for now
    cdef int num_features = 5
    cdef uint32_t num_docs = documents.size()
    cdef float[:, :] features = np.zeros((num_docs, num_features), dtype=np.float32)
    cdef uint64_t i, j, k

    # cdef unordered_set[uint64_t] seen_tokens = unordered_set[uint64_t]()
    # cdef int num_unique_tokens = 0
    # for i in range(query_tokens.size()):
    #     if seen_tokens.find(query_tokens[i]) == seen_tokens.end():
    #         seen_tokens.insert(query_tokens[i])
    #         num_unique_tokens += 1

    cdef vector[float] bm25s_body = vector[float]()
    for i in range(num_docs):
        bm25s_body.push_back(0.0)
    cdef vector[float] bm25s_title = vector[float]()
    for i in range(num_docs):
        bm25s_title.push_back(0.0)
    cdef vector[uint32_t] num_query_tokens_in_title = vector[uint32_t]()
    for i in range(num_docs):
        num_query_tokens_in_title.push_back(0)
    cdef uint64_t current_token
    cdef float current_idf
    cdef int current_tf

    with nogil:
        for i in range(query_tokens.size()):
            current_token = query_tokens[i]
            current_idf = log((num_total_docs - doc_freqs[current_token] + 0.5) / (doc_freqs[current_token] + 0.5))

            for j in range(num_docs):
                # title field
                current_tf = 0
                for k in range(documents[j].field_lengths[0]):
                    if documents[j].tokens[k] == current_token:
                        current_tf += 1
                        num_query_tokens_in_title[j] += 1
                bm25s_title[j] += current_idf * ((current_tf * (bm25_k + 1)) / (current_tf + bm25_k * (1 - bm25_bs[0] + bm25_bs[0] * (documents[j].field_lengths[0] / average_field_lengths[0]))))
                
                # body field
                current_tf = 0
                for k in range(documents[j].field_lengths[0], documents[j].field_lengths[0] + documents[j].field_lengths[1]):
                    if documents[j].tokens[k] == current_token:
                        current_tf += 1
                bm25s_body[j] += current_idf * ((current_tf * (bm25_k + 1)) / (current_tf + bm25_k * (1 - bm25_bs[1] + bm25_bs[1] * (documents[j].field_lengths[1] / average_field_lengths[1]))))

        # transfer bm25s to features matrix
        for j in range(num_docs):
            features[j, 0] = bm25s_title[j]
            features[j, 1] = bm25s_body[j]
            features[j, 2] = documents[j].field_lengths[0]
            features[j, 3] = documents[j].field_lengths[1]
            features[j, 4] = 1.0 if num_query_tokens_in_title[j] == query_tokens.size() else 0.0
        

        

    return features