from whoosh.analysis import RegexTokenizer
from whoosh.analysis import StopFilter

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, text, is_query=False):
        tokenizer = RegexTokenizer() # erstellt Tokenizer-Objekt aus der bib whoosh                
        token_stream = tokenizer(text) # erzeugt Token-Stream aus dem Text
        
        def StoppFilter(tokens):
            stopper = StopFilter()
            for t in stopper(tokens):
                yield t

        def LowercaseFilter(tokens):
            for t in tokens:
                t.text = t.text.lower()
                yield t

        token_stream = LowercaseFilter(token_stream)
        if not is_query:
            token_stream = StoppFilter(token_stream)
        return [t.text for t in token_stream] # es wird nur der Text gespeichert, ohne andere Metadaten
        
    def tokenize_document(self, document):
        tokens = self.tokenize(document.body)
        tokens.extend(self.tokenize(document.title))
        return tokens

    def tokenize_query(self, query):
        return self.tokenize(query, is_query=True)