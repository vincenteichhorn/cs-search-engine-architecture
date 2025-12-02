from sea.tokenizer import Tokenizer


def test_tokenizer(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("data"))
    tokenizer = Tokenizer(tmp_path)

    text = "hello, world! this is a test."
    tokens, char_pos = tokenizer.py_tokenize(text.encode("utf-8"), is_query=False)
    print(tokens)

    assert tokens == ["hello", "world", "test"]
    assert char_pos == [0, 7, 24]

    query = 'testing and (the tokenizer) with "special" characters!'
    tokens, char_pos = tokenizer.py_tokenize(query.encode("utf-8"), is_query=True)
    print(tokens)
    assert tokens == [
        "test",
        "(",
        "the",
        "token",
        ")",
        "with",
        '"',
        "special",
        '"',
        "charact",
    ] or tokens == [
        "testing",
        "(",
        "the",
        "tokenizer",
        ")",
        "with",
        '"',
        "special",
        '"',
        "characters",
    ]

    assert char_pos == [0, 12, 13, 17, 26, 28, 33, 34, 41, 43]

    assert tokenizer.py_get(0) == "hello"
    assert tokenizer.py_get(1) == "world"
