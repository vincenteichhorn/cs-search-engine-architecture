import mmap
import os
import struct
from sea.posting_list import PostingList
from libc.stdint cimport uint8_t, uint64_t
from sea.posting import deserialize_for_scoring, posting_serialize
from tqdm import tqdm
import heapq
from array import array
from cpython.unicode cimport PyUnicode_DecodeUTF8
from libc.time cimport clock, CLOCKS_PER_SEC, clock_t
import math

cdef int get_part_id(str part_dir):
    return int(part_dir.split("part")[-1])

cdef int pst_id_key(object pst):
    return pst.doc_id

cdef tuple get_next_term_entry(const uint8_t[:] posting_index_view, int view_offset=0):
    cdef int cur = view_offset
    cdef const uint8_t* tmp_ptr = &posting_index_view[0]
    if cur + 4 > posting_index_view.shape[0]:
        return "", b"", 0, 0, 0
    cdef int token_length = (posting_index_view[cur] << 24) | (posting_index_view[cur+1] << 16) | (posting_index_view[cur+2] << 8) | posting_index_view[cur+3]
    cur += 4
    cdef bytes token_bytes = bytes(posting_index_view[cur:cur+token_length])
    cdef str token_str = PyUnicode_DecodeUTF8(<char*>(tmp_ptr + cur), token_length, NULL)
    cur += token_length
    cdef unsigned long long offset = (<uint64_t>posting_index_view[cur] << 56) | (<uint64_t>posting_index_view[cur+1] << 48) | (<uint64_t>posting_index_view[cur+2] << 40) | (<uint64_t>posting_index_view[cur+3] << 32) | (<uint64_t>posting_index_view[cur+4] << 24) | (<uint64_t>posting_index_view[cur+5] << 16) | (<uint64_t>posting_index_view[cur+6] << 8) | (<uint64_t>posting_index_view[cur+7])
    cur += 8
    cdef int length = (posting_index_view[cur] << 24) | (posting_index_view[cur+1] << 16) | (posting_index_view[cur+2] << 8) | posting_index_view[cur+3]
    cur += 4
    return token_str, token_bytes, offset, length, cur - view_offset

cdef class Indexer:

    cdef public object config
    cdef public str save_dir
    cdef public object index
    cdef public list documents
    cdef public int num_total_documents
    cdef public int num_total_postings
    cdef public int partition_id

    cdef public str documents_file_name
    cdef public str document_index_file_name
    cdef public str posting_lists_file_name
    cdef public str posting_lists_index_file_name

    cdef public unsigned int[:] summed_field_lengths
    cdef public dict global_doc_freqs

    cdef public int num_tiers
    cdef public list tier_score_thresholds

    cdef int num_flields
    cdef int partition_size

    cdef public unsigned long long document_index_offset

    def __init__(self, object config=None):
        self.config = config
        self.index = {}
        self.documents = []
        self.num_total_documents = 0
        self.num_total_postings = 0
        self.partition_id = 0
        self.document_index_offset = 0

        self.num_flields = self.config.NUM_FIELDS
        self.partition_size = self.config.PARTITION_SIZE
        self.summed_field_lengths = array('I', [0] * self.num_flields)
        self.global_doc_freqs = {}


        # Clean up existing files
        os.makedirs(self.config.INDEX_PATH, exist_ok=True)

        # Remove old document data/index files
        if self.config.REINDEX_DOCUMENTS:
            for fname in [
                self.config.DOCUMENTS_DATA_FILE_NAME,
                self.config.DOCUMENTS_INDEX_FILE_NAME
            ]:
                fpath = os.path.join(self.config.INDEX_PATH, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)

        # Remove old partition dirs
        # for item in os.listdir(self.config.INDEX_PATH):
        #     item_path = os.path.join(self.config.INDEX_PATH, item)
        #     if os.path.isdir(item_path) and (item.startswith(self.config.PARTITION_PREFIX) or item.startswith(self.config.TIER_PREFIX)):
        #         for file in os.listdir(item_path):
        #             os.remove(os.path.join(item_path, file))
        #         os.rmdir(item_path)

    cpdef add_documents(self, object documents):
        cdef object document
        for document in tqdm(
            documents,
            desc="Indexing",
            total=self.config.MAX_DOCUMENTS,
        ):
            self.add_document(document)
            

    cpdef add_document(self, object document):
        cdef str token
        cdef object posting
        cdef object posting_list
        cdef int doc_freq

        if self.config.REINDEX_DOCUMENTS:
            self.documents.append(document)

        for token in document.get_tokens_unique():
            posting_list = self.index.get(token)
            if posting_list is None:
                posting_list = PostingList(key=pst_id_key)
                self.index[token] = posting_list
            posting = document.get_posting(token)
            for i in range(self.num_flields):
                self.summed_field_lengths[i] += posting.field_lengths[i]
            posting_list.add(posting)
            self.num_total_postings += 1
            doc_freq = self.global_doc_freqs.get(token, 0)
            self.global_doc_freqs[token] = doc_freq + 1
        self.num_total_documents += 1

        if self.num_total_documents % self.partition_size == 0 and self.num_total_documents > 0:
            self.flush()
            self.index = {}
            self.documents = []

    cpdef flush(self):
        cdef str part_dir = os.path.join(self.config.INDEX_PATH, f"{self.config.PARTITION_PREFIX}{self.partition_id}")
        os.makedirs(part_dir, exist_ok=True)
        self.partition_id += 1

        cdef object doc_file, doc_index_file
        cdef object doc
        cdef bytes doc_bytes
        if self.config.REINDEX_DOCUMENTS:
            
            with (
                open(os.path.join(self.config.INDEX_PATH, self.config.DOCUMENTS_DATA_FILE_NAME), "ab") as doc_file,
                open(os.path.join(self.config.INDEX_PATH, self.config.DOCUMENTS_INDEX_FILE_NAME), "ab") as doc_index_file
            ):
                for doc in self.documents:
                    doc_bytes = doc.serialize()
                    doc_file.write(doc_bytes)
                    doc_index_file.write(struct.pack(">Q", self.document_index_offset))
                    doc_index_file.write(struct.pack(">I", len(doc_bytes)))
                    self.document_index_offset += len(doc_bytes)

        cdef list sorted_tokens = sorted(self.index.keys())
        cdef str token
        cdef object posting
        cdef bytes posting_bytes
        cdef int length
        cdef object posting_file, posting_index_file
        cdef unsigned long long offset = 0  

        with (
            open(os.path.join(part_dir, self.config.POSTINGS_DATA_FILE_NAME), "wb") as posting_file,
            open(os.path.join(part_dir, self.config.POSTINGS_INDEX_FILE_NAME), "wb") as posting_index_file
        ):
            for token in sorted_tokens:
                token_bytes = token.encode("utf-8")
                length = 0
                for posting in self.index[token]:
                    posting_bytes = posting_serialize(posting)
                    posting_file.write(posting_bytes)
                    length += len(posting_bytes)
                posting_index_file.write(struct.pack(">I", len(token_bytes)))
                posting_index_file.write(token_bytes)
                posting_index_file.write(struct.pack(">Q", offset))
                posting_index_file.write(struct.pack(">I", length))
                offset += length

    cpdef int get_tier_index(self, float score):
        for i in range(self.config.NUM_TIERS):
            if score >= self.config.TIER_SCORE_THRESHOLDS[i]:
                return i
        return self.config.NUM_TIERS - 1


    cpdef merge_partitions(self):

        with open(os.path.join(self.config.INDEX_PATH, "index_meta.json"), "w") as meta_file:
            import json
            meta = {
                "num_total_documents": self.num_total_documents,
                "num_total_postings": self.num_total_postings,
                "summed_field_lengths": list(self.summed_field_lengths),
                "global_doc_freqs": self.global_doc_freqs
            }
            json.dump(meta, meta_file, indent=4)

        if len(self.index) > 0:
            self.flush()
            self.index = {}
            self.documents = []

        cdef list partition_dirs = [
            os.path.join(self.config.INDEX_PATH, d)
            for d in os.listdir(self.config.INDEX_PATH)
            if os.path.isdir(os.path.join(self.config.INDEX_PATH, d)) and d.startswith(self.config.PARTITION_PREFIX)
        ]
        partition_dirs.sort(key=get_part_id)

        cdef list posting_lists_files = []
        cdef list posting_list_mmaps = []
        cdef list posting_indices_files = []
        cdef list posting_indices_mmaps = []
        cdef list posting_indices_offsets = [0] * len(partition_dirs)
        cdef object pl_file, pi_file
        for part_dir in partition_dirs:
            pl_file = open(os.path.join(part_dir, self.config.POSTINGS_DATA_FILE_NAME), "rb")
            posting_lists_files.append(pl_file)
            pi_file = open(os.path.join(part_dir, self.config.POSTINGS_INDEX_FILE_NAME), "rb")
            posting_indices_files.append(pi_file)

        cdef list tier_posting_list_files = []
        cdef list tier_posting_index_files = []
        cdef list tier_offsets = []
        cdef list tier_lengths = [0] * self.config.NUM_TIERS
        cdef str tier_dir
        for tier in range(self.config.NUM_TIERS):
            tier_dir = os.path.join(self.config.INDEX_PATH, f"{self.config.TIER_PREFIX}{tier}")
            os.makedirs(tier_dir, exist_ok=True)
            pf_file = open(os.path.join(tier_dir, self.config.POSTINGS_DATA_FILE_NAME), "wb")
            pi_file = open(os.path.join(tier_dir, self.config.POSTINGS_INDEX_FILE_NAME), "wb")
            tier_posting_list_files.append(pf_file)
            tier_posting_index_files.append(pi_file)
            tier_offsets.append(0)

        cdef list candidate_partitions = []
        cdef object entry
        cdef int pid
        cdef str token_str
        cdef bytes token_bytes
        cdef unsigned long long offset
        cdef int length
        cdef int bytes_read
        cdef const uint8_t[:] tmp_view

        for pid in range(len(partition_dirs)):
            tmp_view = mmap.mmap(posting_indices_files[pid].fileno(), 0, access=mmap.ACCESS_READ)
            token_str, token_bytes, offset, length, bytes_read = get_next_term_entry(tmp_view, posting_indices_offsets[pid])
            posting_indices_offsets[pid] += bytes_read
            heapq.heappush(candidate_partitions, (token_str, token_bytes, pid, offset, length))

        cdef bytes current_token_bytes = candidate_partitions[0][1]
        cdef list average_field_lengths = [l/self.num_total_documents for l in self.summed_field_lengths]
        cdef int tmp_cur = 0
        cdef object tmp_field_freqs
        cdef object tmp_field_lengths
        cdef bytearray current_postinglist_bytes
        cdef int tmp_posting_length
        cdef int tmp_tier = 0
        cdef float idf
        cdef bytes score_bytes
        cdef int tmp_partition
        cdef int i

        cdef object pbar = tqdm(total=self.num_total_postings, desc="Merging")
        while candidate_partitions:
            token_str, token_bytes, pid, offset, length = candidate_partitions[0]

            if token_bytes != current_token_bytes or pid == -1:

                for tier in range(self.config.NUM_TIERS):
                    if tier_lengths[tier] > 0:
                        tier_posting_index_files[tier].write(struct.pack(">I", len(current_token_bytes)))
                        tier_posting_index_files[tier].write(current_token_bytes)
                        tier_posting_index_files[tier].write(struct.pack(">Q", tier_offsets[tier]))
                        tier_posting_index_files[tier].write(struct.pack(">I", tier_lengths[tier]))
                        tier_posting_index_files[tier].write(struct.pack(">I", df))
                        tier_offsets[tier] += tier_lengths[tier]
                        tier_lengths[tier] = 0
            
                if pid == -1:
                    # dummy entry, all done
                    break

            token_str, token_bytes, pid, offset, length = heapq.heappop(candidate_partitions)
            current_token_bytes = token_bytes

            tmp_cur = 0
            df = self.global_doc_freqs.get(token_str, 1)
            tmp_view = mmap.mmap(posting_lists_files[pid].fileno(), 0, access=mmap.ACCESS_READ)
            current_postinglist_bytes = bytearray(tmp_view[offset:offset+length])
            idf = math.log((self.num_total_documents - df + 0.5)/(df + 0.5))
            while tmp_cur < length:
                tmp_field_freqs, tmp_field_lengths, tmp_posting_length = deserialize_for_scoring(current_postinglist_bytes, tmp_cur)

                tf = 0.0
                for i in range(self.num_flields):
                    tf += self.config.FIELD_BOOSTS[i] * (tmp_field_freqs[i])/(1 - self.config.BM25_B_VALUES[i] + self.config.BM25_B_VALUES[i] * tmp_field_lengths[i]/average_field_lengths[i])

                bm25 = idf * (tf * (self.config.BM25_K + 1))/(tf + self.config.BM25_K)
                if bm25 < 0.0:
                    bm25 = 0.0

                score_bytes = struct.pack(">f", bm25)
                for i in range(4):
                    current_postinglist_bytes[tmp_cur+4 + i] = score_bytes[i]

                tmp_tier = self.get_tier_index(bm25)
                tier_posting_list_files[tmp_tier].write(
                    current_postinglist_bytes[tmp_cur:tmp_cur+tmp_posting_length]
                )
                tier_lengths[tmp_tier] += tmp_posting_length
                pbar.update(1)

                tmp_cur += tmp_posting_length
    
            tmp_view = mmap.mmap(posting_indices_files[pid].fileno(), 0, access=mmap.ACCESS_READ)
            token_str, token_bytes, offset, length, bytes_read = get_next_term_entry(tmp_view, posting_indices_offsets[pid])
            posting_indices_offsets[pid] += bytes_read
            if token_str == "":
                pid = -1  # mark as done
            heapq.heappush(candidate_partitions, (token_str, token_bytes, pid, offset, length))

        # for i in range(len(partition_dirs)):
        #     posting_lists_files[i].close()
        #     posting_indices_files[i].close()
        #     os.remove(os.path.join(partition_dirs[i], self.config.POSTINGS_DATA_FILE_NAME))
        #     os.remove(os.path.join(partition_dirs[i], self.config.POSTINGS_INDEX_FILE_NAME))
        #     os.rmdir(partition_dirs[i])
