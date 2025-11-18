from sea.spelling_corrector import get_bigrams, SpellingCorrector


def test_get_bigrams():

    assert get_bigrams("hello") == ["$h", "he", "el", "ll", "lo", "o$"]
    assert get_bigrams("a") == ["$a", "a$"]
    assert get_bigrams("") == []


def test_get_candidates():

    tokens = ["hello", "yellow", "yes", "hollow", "held", "deoxyribonucleic"]
    corrector = SpellingCorrector(tokens)
    bigrams = get_bigrams("halo")
    candiates = corrector.get_candidates(bigrams)
    assert candiates == ["held", "hello", "hollow", "yellow"]


def test_get_corrections():

    tokens = ["hello", "yellow", "yes", "hollow", "held", "deoxyribonucleic"]
    corrector = SpellingCorrector(tokens)

    corrections = corrector.get_corrections_all("halo")
    assert corrections == ["hello"]


if __name__ == "__main__":

    test_get_candidates()
