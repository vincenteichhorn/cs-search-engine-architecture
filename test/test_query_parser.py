from sea.tokenizer import Tokenizer
from sea.query import QueryParser


def test_query_parser(tmp_path_factory):
    tmp_dir = str(tmp_path_factory.mktemp("test_query_parser"))

    tokenizer = Tokenizer(tmp_dir)
    tokenizer.py_tokenize(b'and or not ( ) "', True)  # Preload special tokens into vocabulary
    query_parser = QueryParser(
        and_operator=tokenizer.py_vocab_lookup(b"and"),
        or_operator=tokenizer.py_vocab_lookup(b"or"),
        not_operator=tokenizer.py_vocab_lookup(b"not"),
        open_paren=tokenizer.py_vocab_lookup(b"("),
        close_paren=tokenizer.py_vocab_lookup(b")"),
        phrase_marker=tokenizer.py_vocab_lookup(b'"'),
    )

    tokens, _ = tokenizer.py_tokenize(b'apple banana cherry and or not ( ) " berlin wall blockade"', True)
    apple_token = tokenizer.py_vocab_lookup(bytes(tokens[0], "utf-8"))
    banana_token = tokenizer.py_vocab_lookup(bytes(tokens[1], "utf-8"))
    cherry_token = tokenizer.py_vocab_lookup(bytes(tokens[2], "utf-8"))
    and_token = tokenizer.py_vocab_lookup(bytes(tokens[3], "utf-8"))
    or_token = tokenizer.py_vocab_lookup(bytes(tokens[4], "utf-8"))
    not_token = tokenizer.py_vocab_lookup(bytes(tokens[5], "utf-8"))
    open_paren_token = tokenizer.py_vocab_lookup(bytes(tokens[6], "utf-8"))
    close_paren_token = tokenizer.py_vocab_lookup(bytes(tokens[7], "utf-8"))
    phrase_marker_token = tokenizer.py_vocab_lookup(bytes(tokens[8], "utf-8"))
    berlin_token = tokenizer.py_vocab_lookup(bytes(tokens[9], "utf-8"))
    wall_token = tokenizer.py_vocab_lookup(bytes(tokens[10], "utf-8"))
    blockade_token = tokenizer.py_vocab_lookup(bytes(tokens[11], "utf-8"))

    tests = {
        b"and": None,
        b"or": None,
        b"not": None,
        b"apple": {"type": "token", "value": apple_token},
        b"apple and banana": {
            "type": "operator",
            "operator": and_token,
            "left": {"type": "token", "value": apple_token},
            "right": {"type": "token", "value": banana_token},
        },
        b"apple or banana": {
            "type": "operator",
            "operator": or_token,
            "left": {"type": "token", "value": apple_token},
            "right": {"type": "token", "value": banana_token},
        },
        b"not apple": {
            "type": "operator",
            "operator": not_token,
            "left": {},
            "right": {"type": "token", "value": apple_token},
        },
        b"apple banana": {
            "type": "operator",
            "operator": and_token,
            "left": {"type": "token", "value": apple_token},
            "right": {"type": "token", "value": banana_token},
        },
        b"apple and banana or cherry": {
            "type": "operator",
            "operator": or_token,
            "left": {
                "type": "operator",
                "operator": and_token,
                "left": {"type": "token", "value": apple_token},
                "right": {"type": "token", "value": banana_token},
            },
            "right": {"type": "token", "value": cherry_token},
        },
        b"apple or banana and cherry": {
            "type": "operator",
            "operator": or_token,
            "left": {"type": "token", "value": apple_token},
            "right": {
                "type": "operator",
                "operator": and_token,
                "left": {"type": "token", "value": banana_token},
                "right": {"type": "token", "value": cherry_token},
            },
        },
        b"not apple and banana": {
            "type": "operator",
            "operator": and_token,
            "left": {
                "type": "operator",
                "operator": not_token,
                "left": {},
                "right": {"type": "token", "value": apple_token},
            },
            "right": {"type": "token", "value": banana_token},
        },
        b'"apple banana" and cherry': {
            "type": "operator",
            "operator": and_token,
            "left": {
                "type": "phrase",
                "values": [
                    apple_token,
                    banana_token,
                ],
            },
            "right": {"type": "token", "value": cherry_token},
        },
        b"berlin and not (wall and blockade)": {
            "type": "operator",
            "operator": and_token,
            "left": {"type": "token", "value": berlin_token},
            "right": {
                "type": "operator",
                "operator": not_token,
                "left": {},
                "right": {
                    "type": "operator",
                    "operator": and_token,
                    "left": {"type": "token", "value": wall_token},
                    "right": {"type": "token", "value": blockade_token},
                },
            },
        },
        b'"and and"': None,
        b'""': None,
    }

    for query, expected in tests.items():
        tokens, _ = tokenizer.py_tokenize(query, True)
        tokens = [tokenizer.py_vocab_lookup(bytes(t, "utf-8")) for t in tokens]
        parsed_tree = query_parser.py_parse(tokens)
        print(f"Query: {query.decode('utf-8')}")
        print("Parsed tree:", parsed_tree)
        print("Expected tree:", expected)
        assert parsed_tree == expected, f"Failed for query: {query}"
