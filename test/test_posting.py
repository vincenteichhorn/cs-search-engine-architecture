from sea.document import Document
from sea.posting import Posting
from sea.tokenizer import Tokenizer


def test_posting_serialization():

    title = "Test Document"
    url = "http://example.com"
    body = "This is a test document for serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    posting = document.get_posting(document.tokens[0])

    data = posting.serialize()
    deserialized_posting, _ = Posting.deserialize(data)

    assert posting.doc_id == deserialized_posting.doc_id
    assert posting.positions == deserialized_posting.positions


def test_posting_list_serialization():

    title = "Another Test Document"
    url = "http://example.org"
    body = "This document is for testing posting list serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    postings = [document.get_posting(token) for token in document.tokens[:3]]
    data = bytes()
    for posting in postings:
        data += posting.serialize()

    deserialized_postings = []
    remainder = data
    while len(remainder) > 0:
        posting, remainder = Posting.deserialize(remainder)
        deserialized_postings.append(posting)

    assert len(postings) == len(deserialized_postings)
    for original, deserialized in zip(postings, deserialized_postings):
        assert original.doc_id == deserialized.doc_id
        assert original.positions == deserialized.positions


if __name__ == "__main__":
    test_posting_list_serialization()
