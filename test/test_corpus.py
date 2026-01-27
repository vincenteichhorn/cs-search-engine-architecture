from sea.corpus import (
    Corpus,
    py_string_processor,
    py_document_processor,
    py_tokenized_document_processor,
)
from sea.tokenizer import Tokenizer
import pytest


def test_corpus_iter(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 100
    c = 0
    got = []
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_next(py_string_processor)
            got.append(docB)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break

    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_get(c, py_string_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break


def test_corpus_persitence(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_next(py_string_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break
    corpus.flush()
    del corpus
    corpus = Corpus(tmp_path, data)

    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_get(c, py_string_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break


def test_corpus_persitence_mmap(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_next(py_string_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break
    corpus.flush()
    del corpus
    corpus = Corpus(tmp_path, "", mmap=True)

    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_get(c, py_string_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break


def test_corpus_reload_iter(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 10
    docs = []
    for _ in range(MAX_ITER):
        doc = corpus.py_next(py_string_processor)
        docs.append(doc)

    corpus.flush()
    del corpus
    corpus = Corpus(tmp_path, data)

    for _ in range(MAX_ITER):
        doc = corpus.py_next(py_string_processor)
        assert doc not in docs


def test_document_processor(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    def wrapper(id, data, offset, length):
        return py_document_processor(id, data, offset, length, lowercase=True)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            doc = corpus.py_next(wrapper)
            expected_line = str(line, "utf-8").split("\t")
            assert doc["url"] == expected_line[1].lower()
            assert doc["title"] == expected_line[2].lower()
            assert doc["body"] == expected_line[3].lower().rstrip("\r\n")
            c += 1
            if c > MAX_ITER:
                break


def test_tokenized_document_processor(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)
    tokenizer = Tokenizer(tmp_path)

    def wrapper(id, data, offset, length):
        return py_tokenized_document_processor(id, data, offset, length, tokenizer)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            tokenized_document = corpus.py_next(wrapper)
            assert len(tokenized_document["tokens"]) > 0
            assert len(tokenized_document["postings"]) == len(tokenized_document["tokens"])
            c += 1
            if c > MAX_ITER:
                break


if __name__ == "__main__":
    import shutil

    tmp_path = "./tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_next(py_string_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break
    corpus.flush()
    del corpus
    corpus = Corpus(tmp_path, "", mmap=True)

    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.py_get(c, py_document_processor)
            print(docB)
            c += 1
            if c > MAX_ITER:
                break
