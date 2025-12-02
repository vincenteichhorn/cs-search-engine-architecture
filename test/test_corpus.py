from sea.corpus import Corpus, identity_processor, document_processor, tokenized_document_processor
from sea.tokenizer import Tokenizer
import pytest


def test_corpus_iter(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            _, docB = corpus.next(identity_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break
    c = 0
    with open(data, "rb") as f:
        for line in f:
            docB = corpus.get(c, identity_processor)
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
            _, docB = corpus.next(identity_processor)
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
            docB = corpus.get(c, identity_processor)
            assert str(line, "utf-8") == docB
            c += 1
            if c > MAX_ITER:
                break


def test_corpus_reload_iter(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 10
    ids = []
    docs = []
    for _ in range(MAX_ITER):
        id, doc = corpus.next(identity_processor)
        ids.append(id)
        docs.append(doc)

    corpus.flush()
    del corpus
    corpus = Corpus(tmp_path, data)

    for _ in range(MAX_ITER):
        id, doc = corpus.next(identity_processor)
        print(id)
        assert id not in ids
        assert doc not in docs


def test_document_processor(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            _, doc = corpus.next(document_processor)
            expected_line = str(line, "utf-8").strip().split("\t")
            assert str(doc["url"], "utf-8") == expected_line[1]
            assert str(doc["title"], "utf-8") == expected_line[2]
            assert str(doc["body"], "utf-8") == expected_line[3]
            c += 1
            if c > MAX_ITER:
                break


def test_tokenized_document_processor(tmp_path_factory):

    tmp_path = str(tmp_path_factory.mktemp("corpus_test"))
    data = "./data/msmarco-docs.tsv"
    corpus = Corpus(tmp_path, data)
    tokenizer = Tokenizer(tmp_path)

    def wrapper(id, ptr, offset, length):
        return tokenized_document_processor(id, ptr, offset, length, tokenizer)

    MAX_ITER = 10
    c = 0
    with open(data, "rb") as f:
        for line in f:
            id, tokenized_document = corpus.next(wrapper)
            print(tokenized_document)
            assert tokenized_document["id"] == id
            assert len(tokenized_document["tokens"]) > 0
            assert len(tokenized_document["token_infos"]) == len(tokenized_document["tokens"])
            c += 1
            if c > MAX_ITER:
                break
