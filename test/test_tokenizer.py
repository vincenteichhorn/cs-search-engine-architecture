from sea.tokenizer import Tokenizer


def test_tokenizer(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("data"))
    tokenizer = Tokenizer(tmp_path)

    text = "hello, world! this is a test."
    tokens, char_pos = tokenizer.py_tokenize(text.encode("utf-8"), is_query=False)

    assert tokens == ["hello", "world", "test"]
    assert char_pos == [0, 7, 24]

    query = 'testing and (the tokenizer) with "special" characters!'
    tokens, char_pos = tokenizer.py_tokenize(query.encode("utf-8"), is_query=True)
    assert tokens == [
        "test",
        "and",
        "(",
        "token",
        ")",
        '"',
        "special",
        '"',
        "charact",
    ]
    assert char_pos == [0, 8, 12, 17, 26, 33, 34, 41, 43]

    assert tokenizer.py_get(0) == "hello"
    assert tokenizer.py_get(1) == "world"


def test_tokenizer_mmap(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("data"))
    tokenizer = Tokenizer(tmp_path, mmap=False)

    text = "memory mapping test! this is a long document with some repeated repeated words. and some content to test mapping."
    tokens_mmap, _ = tokenizer.py_tokenize(text.encode("utf-8"), is_query=False)
    tokenizer.flush()

    del tokenizer
    tokenizer = Tokenizer(tmp_path, mmap=True)

    text = "phosphate binding site prediction in proteins."
    tokens, _ = tokenizer.py_tokenize(text.encode("utf-8"), is_query=False)

    del tokenizer
    tokenizer = Tokenizer(tmp_path, mmap=True)
    tokens, _ = tokenizer.py_tokenize(text.encode("utf-8"), is_query=False)
