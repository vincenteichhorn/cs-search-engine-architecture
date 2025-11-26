from sea.document import Document
from sea.posting import Posting
from sea.tokenizer import Tokenizer
from sea.util.gamma import BitReader, BitWriter


def test_posting_serialization():

    title = "Test Document"
    url = "http://example.com"
    body = "This is a test document for serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    posting = document.get_posting(document.tokens[0])
    posting.score = 101

    data = posting.serialize()
    deserialized_posting, _ = Posting.deserialize(data)

    assert posting.doc_id == deserialized_posting.doc_id
    assert posting.positions == list(deserialized_posting.positions)
    assert posting.field_lengths == list(deserialized_posting.field_lengths)
    assert posting.field_freqs == list(deserialized_posting.field_freqs)
    assert posting.score == deserialized_posting.score


def test_posting_list_serialization():

    title = "Another Test Document"
    url = "http://example.org"
    body = "This document is for testing posting list serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    postings = [document.get_posting(token) for token in document.tokens[:10]]
    data = bytes()
    for i, posting in enumerate(postings):
        posting.score = i * 0.67
        posting_bytes = posting.serialize()
        data += posting_bytes

    deserialized_postings = []
    remainder = data
    while len(remainder) > 0:
        posting, bytes_read = Posting.deserialize(remainder)
        remainder = remainder[bytes_read:]
        deserialized_postings.append(posting)

    assert len(postings) == len(deserialized_postings)
    for original, deserialized in zip(postings, deserialized_postings):
        assert original.doc_id == deserialized.doc_id
        assert original.positions == list(deserialized.positions)
        assert original.field_freqs == list(deserialized.field_freqs)
        assert original.field_lengths == list(deserialized.field_lengths)
        assert original.score == deserialized.score
