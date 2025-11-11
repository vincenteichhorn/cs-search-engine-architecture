from sea.document import Document
from sea.posting import Posting
from sea.tokenizer import Tokenizer


def test_posting_serialize():

    title = "Test Document"
    url = "http://example.com"
    body = "This is a test document for serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    posting = document.get_posting(document.tokens[0])

    bytestring = posting.serialize()
    deserialized_posting = Posting.deserialize(bytestring)

    assert posting.doc_id == deserialized_posting.doc_id
    assert posting.positions == deserialized_posting.positions
