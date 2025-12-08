from sea.util.fast_stemmer import FastStemmer
from sea.tokenizer import Tokenizer

if __name__ == "__main__":

    stemmer = FastStemmer()
    words = [b"running", b"jumps", b"easily", b"fairly"]
    for word in words:
        stemmed = stemmer.py_stem(word)
        print(f"Original: {word.decode('utf-8')}, Stemmed: {stemmed.decode('utf-8')}")

    tokenizer = Tokenizer("/tmp/tokenizer_data")
    text = "The quick brown fox jumps over the lazy dog."
    tokens, char_pos = tokenizer.py_tokenize(text.encode("utf-8"), is_query=False)
    print("Tokens:", tokens)
    print("Character Positions:", char_pos)
