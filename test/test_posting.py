from sea.document import Document
from sea.posting import Posting, posting_deserialize, posting_serialize
from sea.tokenizer import Tokenizer


def test_posting_serialization():

    title = "Test Document"
    url = "http://example.com"
    body = "This is a test document for serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    document.ensure_tokenized()
    posting = document.get_posting(document.tokens[0])
    posting.score = 101

    data = posting_serialize(posting)
    deserialized_posting, _ = posting_deserialize(data)

    assert posting.doc_id == deserialized_posting.doc_id
    assert posting.char_positions == list(deserialized_posting.char_positions)
    assert posting.field_lengths == list(deserialized_posting.field_lengths)
    assert posting.field_freqs == list(deserialized_posting.field_freqs)
    assert posting.score == deserialized_posting.score


def test_posting_list_serialization():

    title = "Another Test Document"
    url = "http://example.org"
    body = "This document is for testing posting list serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    document.ensure_tokenized()
    postings = [document.get_posting(token) for token in document.tokens[:10]]
    data = bytes()
    for i, posting in enumerate(postings):
        posting.score = i * 0.67
        posting_bytes = posting_serialize(posting)
        data += posting_bytes

    deserialized_postings = []
    remainder = data
    while len(remainder) > 0:
        posting, bytes_read = posting_deserialize(remainder)
        remainder = remainder[bytes_read:]
        deserialized_postings.append(posting)

    assert len(postings) == len(deserialized_postings)
    for original, deserialized in zip(postings, deserialized_postings):
        assert original.doc_id == deserialized.doc_id
        assert original.char_positions == list(deserialized.char_positions)
        assert original.field_freqs == list(deserialized.field_freqs)
        assert original.field_lengths == list(deserialized.field_lengths)
        assert original.score == deserialized.score
