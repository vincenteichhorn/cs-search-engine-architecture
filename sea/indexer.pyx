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


cdef float K1 = 1.5
cdef list B1s = [0.75, 0.75]
cdef list boosts = [1.0, 0.5]

cdef tuple sort_by(tuple tuple):
    return tuple[1], tuple[0]

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
    token = token_bytes.decode("utf-8")
    offset = struct.unpack(">Q", posting_index_file.read(8))[0]
    length = struct.unpack(">I", posting_index_file.read(4))[0]
    return token, token_bytes, offset, length


cdef class Indexer:

    cdef public int partition_size
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

    def __init__(self, str save_dir, int partition_size=1000):
        self.partition_size = partition_size
        self.save_dir = save_dir
        self.index = {}
        self.documents = []
        self.num_total_documents = 0
        self.num_total_postings = 0
        self.summed_field_lengths = []
        self.partition_id = 0
        self.document_index_offset = 0

        self.num_tiers = 3
        self.tier_score_thresholds = [5, 2, 0]

        self.documents_file_name = "documents.bin"
        self.document_index_file_name = "document_index.bin"
        self.posting_lists_file_name = "posting_lists.bin"
        self.posting_lists_index_file_name = "posting_lists_index.bin"

        # Clean up existing files
        os.makedirs(self.save_dir, exist_ok=True)

        # Remove old main index files
        for fname in [
            self.documents_file_name,
            self.document_index_file_name,
            self.posting_lists_file_name,
            self.posting_lists_index_file_name
        ]:
            fpath = os.path.join(self.save_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        # Remove old partition dirs
        for item in os.listdir(self.save_dir):
            item_path = os.path.join(self.save_dir, item)
            if os.path.isdir(item_path) and (item.startswith("part") or item.startswith("tier")):
                for file in os.listdir(item_path):
                    os.remove(os.path.join(item_path, file))
                os.rmdir(item_path)


    cpdef add_document(self, object document):
        self.documents.append(document)
        cdef str token
        cdef object posting
        cdef set tokens = set(document.tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = PostingList(key=pst_id_key)
            posting = document.get_posting(token)
            if len(self.summed_field_lengths) == 0:
                self.summed_field_lengths = posting.field_lengths[:]
            else:
                for i in range(len(posting.field_lengths)):
                    self.summed_field_lengths[i] += posting.field_lengths[i]
            self.index[token].add(posting)
            self.num_total_postings += 1
        self.num_total_documents += 1

        if self.num_total_documents % self.partition_size == 0 and self.num_total_documents > 0:
            self.flush()
            self.index = {}
            self.documents = []

    cpdef flush(self):
        cdef str part_dir = os.path.join(self.save_dir, f"part{self.partition_id}")
        os.makedirs(part_dir, exist_ok=True)
        self.partition_id += 1

        cdef object doc_file, doc_index_file
        cdef object doc
        cdef bytes doc_bytes
        with (
            open(os.path.join(self.save_dir, self.documents_file_name), "ab") as doc_file,
            open(os.path.join(self.save_dir, self.document_index_file_name), "ab") as doc_index_file
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
            open(os.path.join(part_dir, self.posting_lists_file_name), "wb") as posting_file,
            open(os.path.join(part_dir, self.posting_lists_index_file_name), "wb") as posting_index_file
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
        for i in range(self.num_tiers):
            if score >= self.tier_score_thresholds[i]:
                return i
        return self.num_tiers - 1

    cpdef merge_partitions(self):

        if len(self.index) > 0:
            self.flush()
            self.index = {}
            self.documents = []

        cdef list partition_dirs = [
            os.path.join(self.save_dir, d)
            for d in os.listdir(self.save_dir)
            if os.path.isdir(os.path.join(self.save_dir, d)) and d.startswith("part")
        ]
        partition_dirs.sort(key=get_part_id)

        cdef list posting_lists_files = []
        cdef list posting_indices = []
        cdef list posting_list_mmaps = []
        cdef list posting_list_views = []
        cdef object pl_file, pi_file, mm
        cdef const uint8_t[:] view
        for part_dir in partition_dirs:
            pl_file = open(os.path.join(part_dir, self.posting_lists_file_name), "rb")
            pi_file = open(os.path.join(part_dir, self.posting_lists_index_file_name), "rb")
            mm = mmap.mmap(pl_file.fileno(), 0, access=mmap.ACCESS_READ)
            view = mm
            posting_lists_files.append(pl_file)
            posting_indices.append(pi_file)
            posting_list_mmaps.append(mm)
            posting_list_views.append(view)

        
        cdef list tier_posting_list_files = []
        cdef list tier_posting_index_files = []
        cdef list tier_offsets = []
        cdef list tier_lengths = []
        cdef str tier_dir
        for tier in range(self.num_tiers):
            tier_dir = os.path.join(self.save_dir, f"tier{tier}")
            os.makedirs(tier_dir, exist_ok=True)
            pf_file = open(os.path.join(tier_dir, self.posting_lists_file_name), "wb")
            pi_file = open(os.path.join(tier_dir, self.posting_lists_index_file_name), "wb")
            tier_posting_list_files.append(pf_file)
            tier_posting_index_files.append(pi_file)
            tier_offsets.append(0)
            tier_lengths.append(0)


        cdef list candidate_partitions = []
        cdef object entry
        cdef int pid
        cdef str token
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

        with (
            open(os.path.join(self.save_dir, self.posting_lists_file_name), "wb") as merged_posting_file,
            open(os.path.join(self.save_dir, self.posting_lists_index_file_name), "wb") as merged_posting_index_file
        ):
            for pid, posting_list_index in enumerate(posting_indices):
                entry = get_next_term_entry(posting_list_index)
                if entry:
                    token, token_bytes, offset, length = entry
                    candidate_partitions.append((pid, token, token_bytes, offset, length))

            candidate_partitions.sort(key=sort_by)
            current_token, current_token_bytes = candidate_partitions[0][1], candidate_partitions[0][2]

            while candidate_partitions:
                pid, token, token_bytes, offset, length = candidate_partitions.pop(0)
                if token != current_token:
                    for pst in current_postings:
                        field_token_frequencies = pst.field_freqs
                        field_lengths = pst.field_lengths
                        score = fielded_bm25(field_token_frequencies, field_lengths, average_field_lengths, len(current_postings), self.num_total_documents, K1, B1s, boosts)
                        pst.score = score
                        tier_idx = self.get_tier_index(score)

                        tmp_posting_bytes = pst.serialize()
                        tier_posting_list_files[tier_idx].write(tmp_posting_bytes)
                        tier_lengths[tier_idx] += len(tmp_posting_bytes)
                        pbar.update(1)

                    for tier in range(self.num_tiers):
                        if tier_lengths[tier] == 0:
                            continue
                        tier_posting_index_files[tier].write(struct.pack(">I", len(current_token_bytes)))
                        tier_posting_index_files[tier].write(current_token_bytes)
                        tier_posting_index_files[tier].write(struct.pack(">Q", tier_offsets[tier]))
                        tier_posting_index_files[tier].write(struct.pack(">I", tier_lengths[tier]))
                        tier_posting_index_files[tier].write(struct.pack(">I", len(current_postings)))
                        tier_offsets[tier] += tier_lengths[tier]
                        tier_lengths[tier] = 0

                    current_postings = []
                    current_token, current_token_bytes = token, token_bytes

                tmp_end_offset = offset + length
                tmp_cur = offset
                while tmp_cur < tmp_end_offset:
                    posting, bytes_read = posting_deserialize(posting_list_views[pid][tmp_cur:tmp_end_offset], only_doc_id=False)
                    tmp_cur += bytes_read
                    current_postings.append(posting)

                entry = get_next_term_entry(posting_indices[pid])
                if entry:
                    t, t_bytes, o, l = entry
                    # binary insert
                    left, right = 0, len(candidate_partitions) - 1
                    while left <= right:
                        mid = (left + right) // 2
                        if (t, pid) < (candidate_partitions[mid][1], candidate_partitions[mid][0]):
                            right = mid - 1
                        else:
                            left = mid + 1
                    candidate_partitions.insert(left, (pid, t, t_bytes, o, l))

        # TODO write last current_token index infos

        cdef int i
        for i in range(len(partition_dirs)):
            posting_lists_files[i].close()
            posting_indices[i].close()
            part_dir = partition_dirs[i]
            os.remove(os.path.join(part_dir, self.posting_lists_file_name))
            os.remove(os.path.join(part_dir, self.posting_lists_index_file_name))
            os.rmdir(part_dir)
