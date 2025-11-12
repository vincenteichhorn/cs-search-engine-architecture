from sea.query import Node, Query
from sea.tokenizer import Tokenizer


def test_query():
    tokenizer = Tokenizer()
    tests = {
        "and": None,
        "or": None,
        "not": None,
        "the": None,
        "apple": Node("apple"),
        "apple and banana": Node("and", left=Node("apple"), right=Node("banana")),
        "apple or banana": Node("or", left=Node("apple"), right=Node("banana")),
        "not apple": Node("not", right=Node("apple")),
        "apple banana": Node("and", left=Node("apple"), right=Node("banana")),
        "apple and banana or cherry": Node(
            "or",
            left=Node("and", left=Node("apple"), right=Node("banana")),
            right=Node("cherry"),
        ),
        "apple or banana and cherry": Node(
            "or",
            left=Node("apple"),
            right=Node("and", left=Node("banana"), right=Node("cherry")),
        ),
        "not apple and banana": Node(
            "and",
            left=Node("not", right=Node("apple")),
            right=Node("banana"),
        ),
        '"apple banana" and cherry': Node(
            "and",
            left=Node(["apple", "banana"]),
            right=Node("cherry"),
        ),
        '"and and"': None,
        '""': None,
    }
    for query_str, expected_tree in tests.items():
        query = Query(query_str, tokenizer)
        if expected_tree is None:
            assert query.root is None, f"Failed for query: {query_str}"
        else:
            assert repr(query.root) == repr(expected_tree), f"Failed for query: {query_str}"
