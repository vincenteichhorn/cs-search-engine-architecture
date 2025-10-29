from typing import List
from whoosh.analysis import RegexTokenizer
from nltk.stem import PorterStemmer
from sea.document import Document


class Tokenizer:
    def __init__(self, stop_words=None):

        self.stop_words = stop_words or (
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "can",
            "for",
            "from",
            "have",
            "if",
            "in",
            "is",
            "it",
            "may",
            "not",
            "of",
            "on",
            "or",
            "tbd",
            "that",
            "the",
            "this",
            "to",
            "us",
            "we",
            "when",
            "will",
            "with",
            "yet",
            "you",
            "your",
        )
        self.query_stop_words = set(self.stop_words).difference(set(["and", "or", "not"]))  # Can be extended for query-specific stop words

    def tokenize(self, text: str, is_query: bool = False) -> List[str]:
        """
        Tokenize the input text into a stream of tokens.

        Args:
            text (str): The input text to be tokenized.
            is_query (bool, optional): If True, do not remove stop words (default is False).

        Returns:
            List[str]: A list of token strings extracted from the input text.
        """
        tokenizer = RegexTokenizer()
        token_stream = tokenizer(text)

        def StopFilter(tokens, stop_words):
            """
            Filter out stop words from the token stream. Uses a predefined list of common English stop words.
            """
            for t in tokens:
                if t.text not in stop_words:
                    yield t

        def LowercaseFilter(tokens):
            """
            Convert all tokens in the token stream to lowercase.
            """
            for t in tokens:
                t.text = t.text.lower()
                yield t

        def StemmerFilter(tokens):
            """
            Apply Porter stemming to each token in the token stream.
            """
            stemmer = PorterStemmer()
            for t in tokens:
                t.text = stemmer.stem(t.text)
                yield t

        token_stream = LowercaseFilter(token_stream)
        if is_query:
            token_stream = StopFilter(token_stream, self.query_stop_words)
        else:
            token_stream = StopFilter(token_stream, self.stop_words)

        token_stream = StemmerFilter(token_stream)
        return [t.text for t in token_stream]

    def tokenize_document(self, document: Document) -> List[str]:
        """
        Tokenize the content of a document, including its body and title.

        Args:
            document (Document): The Document object whose content is to be tokenized.

        Returns:
            List[str]: A list of token strings extracted from the document's content.
        """
        tokens = self.tokenize(document.body)
        tokens.extend(self.tokenize(document.title))
        return tokens

    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize a search query.

        Args:
            query (str): The search query to be tokenized.

        Returns:
            List[str]: A list of token strings extracted from the search query.
        """
        return self.tokenize(query, is_query=True)
