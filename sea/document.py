from collections import defaultdict


class Document:
    _next_id = 1

    def __init__(self, title, url, body, tokenizer):
        self.title = title
        self.url = url
        self.body = body
        self.id = Document._next_id
        self.tokens = None
        self.tokenizer = tokenizer
        self._tokenize()
        self.token_counts = None
        self._count_tokens()
        Document._next_id += 1

    def __repr__(self):
        return f"Document(id={self.id}, title={self.title}, url={self.url})"
    
    def _tokenize(self):
        """
        ruft den Tokenizer auf, um das Dokument zu tokenisieren
        """
        if self.tokens is None:
            self.tokens = self.tokenizer.tokenize_document(self)

    def _count_tokens(self):
        """
        Zählt die Häufigkeit jedes Tokens im Dokument
        """
        if self.token_counts is None and self.tokens is not None:
            counts = defaultdict(int)
            for token in self.tokens:
                counts[token] += 1
            self.token_counts = counts
        else:
            raise ValueError("Document must be tokenized before counting tokens.")