from sea.util import encode_gamma, decode_gamma, pack_gammas, unpack_gammas


def test_gamma_codes():

    numbers = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]
    for number in numbers:
        encoded = encode_gamma(number)
        decoded, length = decode_gamma(encoded)
        assert decoded == number


def test_pack_unpack_gammas():

    numbers = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]
    packed = pack_gammas(numbers)
    unpacked = unpack_gammas(packed)
    assert unpacked == numbers
