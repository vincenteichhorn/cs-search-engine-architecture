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
    postings = [document.get_posting(token) for token in document.tokens[:10]]
    data = bytes()
    for posting in postings:
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
        assert original.positions == deserialized.positions


def test_pack_unpack_gammas_postings():

    postings = (Posting(1, [0, 2, 4]), Posting(3, [1, 3]), Posting(5, [0, 1, 2, 3]))
    writer = BitWriter()
    for posting in postings:
        posting.serialize_gamma(writer)

    data = writer.get_bytes()
    reader = BitReader(bytes(data))
    unpacked_postings = []
    while reader.bits_remaining() > 0:
        posting = Posting.deserialize_gamma(reader)
        unpacked_postings.append(posting)
    for p1, p2 in zip(postings, unpacked_postings):
        assert p1.doc_id == p2.doc_id
        assert p1.positions == p2.positions
