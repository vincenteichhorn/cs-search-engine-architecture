from sea.tokenizer import Tokenizer


def test_tokenizer(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("data"))
    tokenizer = Tokenizer(tmp_path)

    text = "Hello, World! This is a test."
    tokens, char_pos = tokenizer.py_tokenize(text.encode("utf-8"), is_query=False)

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
