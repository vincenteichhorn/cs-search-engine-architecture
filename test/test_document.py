from sea.document import Document
from sea.tokenizer import Tokenizer


def test_document_serialize():

    title = "Test Document"
    url = "http://example.com"
    body = "This is a test document for serialization."
    tokenizer = Tokenizer()
    document = Document(title, url, body, tokenizer)
    document.ensure_tokenized()

    bytestring = document.serialize()
    deserialized_document = Document.deserialize(bytestring)

    assert document.id == deserialized_document.id
    assert document.title == deserialized_document.title
    assert document.url == deserialized_document.url
    assert document.body == deserialized_document.body
