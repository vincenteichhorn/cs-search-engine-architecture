from sea.query import Node, Query
from sea.tokenizer import Tokenizer


def test_query():
    tokenizer = Tokenizer()
    tests = {
        "and": None,
        "or": None,
        "not": None,
        "the": None,
        "apple": Node("appl"),
        "apple and banana": Node("and", left=Node("appl"), right=Node("banana")),
        "apple or banana": Node("or", left=Node("appl"), right=Node("banana")),
        "not apple": Node("not", right=Node("appl")),
        "apple banana": Node("and", left=Node("appl"), right=Node("banana")),
        "apple and banana or cherry": Node(
            "or",
            left=Node("and", left=Node("appl"), right=Node("banana")),
            right=Node("cherri"),
        ),
        "apple or banana and cherry": Node(
            "or",
            left=Node("appl"),
            right=Node("and", left=Node("banana"), right=Node("cherri")),
        ),
        "not apple and banana": Node(
            "and",
            left=Node("not", right=Node("appl")),
            right=Node("banana"),
        ),
        '"apple banana" and cherry': Node(
            "and",
            left=Node(["appl", "banana"]),
            right=Node("cherri"),
        ),
        '"and and"': None,
        '""': None,
    }
    for query_str, expected_tree in tests.items():
        query = Query(query_str, tokenizer)
        if expected_tree is None:
            assert query.root is None, f"Failed for query: {query_str}"
        else:
            print(f"Testing query: {query_str}")
            print(f"Expected tree: {expected_tree}")
            print(f"Actual tree:   {query.root}")
            assert repr(query.root) == repr(expected_tree), f"Failed for query: {query_str}"
