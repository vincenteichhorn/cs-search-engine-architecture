class Document:
    _next_id = 1

    def __init__(self, title, url, body):
        self.title = title
        self.url = url
        self.body = body
        self.id = Document._next_id
        self.tokens = None
        Document._next_id += 1

    def __repr__(self):
        return f"Document(id={self.id}, title={self.title}, url={self.url})"