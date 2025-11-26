from cpython.dict cimport PyDict_New
import mmap
import os
import struct
from sea.document import Document
from sea.posting_list import PostingList
from libc.stdint cimport uint8_t, uint64_t
from sea.posting import posting_deserialize
from sea.bm25 import fielded_bm25
from tqdm import tqdm
import heapq


cdef int get_part_id(str part_dir):
    return int(part_dir.split("part")[-1])

cdef int pst_id_key(object pst):
    return pst.doc_id

cdef object get_next_term_entry(object posting_index_file):
    data = posting_index_file.read(4)
    if not data:
        return None
    token_length = struct.unpack(">I", data)[0]
    token_bytes = posting_index_file.read(token_length)
    # token = token_bytes.decode("utf-8")
    offset = struct.unpack(">Q", posting_index_file.read(8))[0]
    length = struct.unpack(">I", posting_index_file.read(4))[0]
    return token_bytes, offset, length


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
    cdef public list summed_field_lengths

    cdef public int num_tiers
    cdef public list tier_score_thresholds

    cdef public unsigned long long document_index_offset

    def __init__(self, object config=None):
        self.config = config
        self.index = {}
        self.documents = []
        self.num_total_documents = 0
        self.num_total_postings = 0
        self.partition_id = 0
        self.document_index_offset = 0

        self.summed_field_lengths = [0] * self.config.NUM_FIELDS

        # Clean up existing files
        os.makedirs(self.config.INDEX_PATH, exist_ok=True)

        # Remove old document data/index files
        for fname in [
            self.config.DOCUMENTS_DATA_FILE_NAME,
            self.config.DOCUMENTS_INDEX_FILE_NAME
        ]:
            fpath = os.path.join(self.config.INDEX_PATH, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        # Remove old partition dirs
        for item in os.listdir(self.config.INDEX_PATH):
            item_path = os.path.join(self.config.INDEX_PATH, item)
            if os.path.isdir(item_path) and (item.startswith(self.config.PARTITION_PREFIX) or item.startswith(self.config.TIER_PREFIX)):
                for file in os.listdir(item_path):
                    os.remove(os.path.join(item_path, file))
                os.rmdir(item_path)

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
        cdef set tokens = set(document.tokens)
        self.documents.append(document)

        for token in tokens:
            if token not in self.index:
                self.index[token] = PostingList(key=pst_id_key)
            posting = document.get_posting(token)
            for i in range(self.config.NUM_FIELDS):
                self.summed_field_lengths[i] += posting.field_lengths[i]
            self.index[token].add(posting)
            self.num_total_postings += 1
        self.num_total_documents += 1

        if self.num_total_documents % self.config.PARTITION_SIZE == 0 and self.num_total_documents > 0:
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
                    posting_bytes = posting.serialize()
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

    
    cpdef list write_term_postings(
        self, 
        list current_postings, 
        bytes current_token_bytes, 
        list tier_posting_list_files, 
        list tier_posting_index_files, 
        list tier_offsets, 
        list average_field_lengths, 
        object pbar
    ):

        cdef list tier_lengths = [0] * self.config.NUM_TIERS
        cdef object pst
        cdef float score
        cdef bytes tmp_posting_bytes
        cdef int tier
        cdef object field_token_frequencies
        cdef object field_lengths

        for pst in current_postings:
            field_token_frequencies = pst.field_freqs
            field_lengths = pst.field_lengths
            score = fielded_bm25(field_token_frequencies, field_lengths, average_field_lengths, len(current_postings), self.num_total_documents, self.config.BM25_K, self.config.BM25_B_VALUES, self.config.FIELD_BOOSTS)
            pst.score = score
            tier = self.get_tier_index(score)

            tmp_posting_bytes = pst.serialize()
            tier_posting_list_files[tier].write(tmp_posting_bytes)
            tier_lengths[tier] += len(tmp_posting_bytes)
            pbar.update(1)

        for tier in range(self.config.NUM_TIERS):
            tier_posting_list_files[tier].flush()
            if tier_lengths[tier] == 0:
                continue
            tier_posting_index_files[tier].write(struct.pack(">I", len(current_token_bytes)))
            tier_posting_index_files[tier].write(current_token_bytes)
            tier_posting_index_files[tier].write(struct.pack(">Q", tier_offsets[tier]))
            tier_posting_index_files[tier].write(struct.pack(">I", tier_lengths[tier]))
            tier_posting_index_files[tier].write(struct.pack(">I", len(current_postings)))
            tier_posting_index_files[tier].flush()
        return tier_lengths


    cpdef merge_partitions(self):

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
        cdef list posting_list_views = []
        cdef list posting_indices_files = []
        cdef object pl_file, pi_file, mm
        cdef const uint8_t[:] view
        for part_dir in partition_dirs:
            pl_file = open(os.path.join(part_dir, self.config.POSTINGS_DATA_FILE_NAME), "rb")
            mm = mmap.mmap(pl_file.fileno(), 0, access=mmap.ACCESS_READ)
            view = mm
            posting_lists_files.append(pl_file)
            posting_list_mmaps.append(mm)
            posting_list_views.append(view)
            pi_file = open(os.path.join(part_dir, self.config.POSTINGS_INDEX_FILE_NAME), "rb")
            posting_indices_files.append(pi_file)


        cdef list tier_posting_list_files = []
        cdef list tier_posting_index_files = []
        cdef list tier_offsets = []
        cdef list tier_lengths
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
        cdef bytes token_bytes
        cdef unsigned long long offset
        cdef int length

        cdef list current_postings = []
        cdef object tmp_posting
        cdef bytes tmp_posting_bytes
        cdef Py_ssize_t tmp_bytes_read
        cdef Py_ssize_t tmp_end_offset
        cdef Py_ssize_t tmp_cur
        cdef int df
        
        cdef list average_field_lengths = [l/self.num_total_documents for l in self.summed_field_lengths]
        cdef object pbar = tqdm(total=self.num_total_postings, desc="Merging")

        for pid, pi_file in enumerate(posting_indices_files):
            entry = get_next_term_entry(pi_file)
            if entry:
                token_bytes, offset, length = entry
                heapq.heappush(candidate_partitions, (token_bytes, pid, offset, length))

        cdef bytes current_token_bytes = candidate_partitions[0][0]

        while candidate_partitions:
            token_bytes, pid, offset, length = heapq.heappop(candidate_partitions)
            if token_bytes != current_token_bytes:
                tier_lengths = self.write_term_postings(
                    current_postings, 
                    current_token_bytes, 
                    tier_posting_list_files, 
                    tier_posting_index_files, 
                    tier_offsets, 
                    average_field_lengths, 
                    pbar
                )

                for tier in range(self.config.NUM_TIERS):
                    tier_offsets[tier] += tier_lengths[tier]

                current_postings = []
                current_token_bytes = token_bytes

            tmp_end_offset = offset + length
            tmp_cur = offset
            while tmp_cur < tmp_end_offset:
                posting, bytes_read = posting_deserialize(posting_list_views[pid][tmp_cur:tmp_end_offset], only_doc_id=False)
                tmp_cur += bytes_read
                current_postings.append(posting)

            entry = get_next_term_entry(posting_indices_files[pid])
            if entry:
                tb, o, l = entry
                heapq.heappush(candidate_partitions, (tb, pid, o, l))

        _ = self.write_term_postings(
            current_postings, 
            current_token_bytes, 
            tier_posting_list_files, 
            tier_posting_index_files, 
            tier_offsets, 
            average_field_lengths, 
            pbar
        )

        cdef int i
        for i in range(len(partition_dirs)):
            posting_lists_files[i].close()
            posting_indices_files[i].close()
            os.remove(os.path.join(partition_dirs[i], self.config.POSTINGS_DATA_FILE_NAME))
            os.remove(os.path.join(partition_dirs[i], self.config.POSTINGS_INDEX_FILE_NAME))
            os.rmdir(partition_dirs[i])
