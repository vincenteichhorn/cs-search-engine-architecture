from cpython.dict cimport PyDict_New
import mmap
import os
import struct
from sea.document import Document
from sea.posting_list import PostingList

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

cdef object write_next_term_entry(object posting_index_file, bytes token_bytes, unsigned long long offset, int length):
    posting_index_file.write(struct.pack(">I", len(token_bytes)))
    posting_index_file.write(token_bytes)
    posting_index_file.write(struct.pack(">Q", offset))
    posting_index_file.write(struct.pack(">I", length))


cdef class Indexer:

    cdef public int partition_size
    cdef public str save_dir
    cdef public object index
    cdef public list documents
    cdef public int no_documents_in_partition
    cdef public int partition_id

    cdef public str documents_file_name
    cdef public str document_index_file_name
    cdef public str posting_lists_file_name
    cdef public str posting_lists_index_file_name

    cdef public unsigned long long document_index_offset

    def __init__(self, str save_dir, int partition_size=1000):
        self.partition_size = partition_size
        self.save_dir = save_dir
        self.index = {}
        self.documents = []
        self.no_documents_in_partition = 0
        self.partition_id = 0

        os.makedirs(self.save_dir, exist_ok=True)

        self.documents_file_name = "documents.bin"
        self.document_index_file_name = "document_index.bin"
        self.posting_lists_file_name = "posting_lists.bin"
        self.posting_lists_index_file_name = "posting_lists_index.bin"

        for fname in [
            self.documents_file_name,
            self.document_index_file_name,
            self.posting_lists_file_name,
            self.posting_lists_index_file_name
        ]:
            fpath = os.path.join(self.save_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)

        self.document_index_offset = 0

        # Remove old partition dirs
        for item in os.listdir(self.save_dir):
            item_path = os.path.join(self.save_dir, item)
            if os.path.isdir(item_path) and item.startswith("part"):
                for file in os.listdir(item_path):
                    os.remove(os.path.join(item_path, file))
                os.rmdir(item_path)

    cpdef add_document(self, object document):
        self.documents.append(document)
        cdef str token
        for token in document.tokens:
            if token not in self.index:
                self.index[token] = PostingList(key=pst_id_key)
            self.index[token].add(document.get_posting(token))
        self.no_documents_in_partition += 1

        if self.no_documents_in_partition >= self.partition_size:
            self.flush()
            self.no_documents_in_partition = 0
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

    cpdef merge_partitions(self):

        if len(self.index) > 0:
            self.flush()

        cdef list partition_dirs = [
            os.path.join(self.save_dir, d)
            for d in os.listdir(self.save_dir)
            if os.path.isdir(os.path.join(self.save_dir, d)) and d.startswith("part")
        ]
        partition_dirs.sort(key=get_part_id)

        cdef list posting_lists_files = []
        cdef list posting_indices = []
        cdef list mmaps = []
        cdef object pl_file, pi_file, mm
        for part_dir in partition_dirs:
            pl_file = open(os.path.join(part_dir, self.posting_lists_file_name), "rb")
            pi_file = open(os.path.join(part_dir, self.posting_lists_index_file_name), "rb")
            size = os.fstat(pl_file.fileno()).st_size
            mm = mmap.mmap(pl_file.fileno(), length=size, access=mmap.ACCESS_READ)
            posting_lists_files.append(pl_file)
            posting_indices.append(pi_file)
            mmaps.append(mm)

        cdef unsigned long long merged_offset = 0
        cdef int merged_length = 0
        cdef list candidate_partitions = []
        cdef object entry
        cdef int pid
        cdef str token
        cdef bytes token_bytes
        cdef unsigned long long offset
        cdef int length

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
                    write_next_term_entry(
                        merged_posting_index_file, current_token_bytes, merged_offset, merged_length
                    )
                    merged_offset += merged_length
                    merged_length = 0
                current_token, current_token_bytes = token, token_bytes

                merged_posting_file.write(mmaps[pid][offset : offset + length])
                merged_length += length

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

        cdef int i
        for i in range(len(partition_dirs)):
            mmaps[i].close()
            posting_lists_files[i].close()
            posting_indices[i].close()
            part_dir = partition_dirs[i]
            os.remove(os.path.join(part_dir, self.posting_lists_file_name))
            os.remove(os.path.join(part_dir, self.posting_lists_index_file_name))
            os.rmdir(part_dir)
