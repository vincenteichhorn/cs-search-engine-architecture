from sea.util.gamma import (
    BitReader,
    BitWriter,
    encode_gamma,
    decode_gamma,
    pack_gammas,
    unpack_gammas,
)


def test_gamma_codes():

    numbers = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]
    for number in numbers:
        writer = BitWriter()
        encode_gamma(writer, number)
        encoded = writer.get_bytes()
        print(number, bin(int.from_bytes(encoded, "big")))
        reader = BitReader(encoded)
        decoded, _ = decode_gamma(reader)
        assert decoded == number


def test_pack_unpack_gammas():

    numbers = [1, 2, 3]
    packed = pack_gammas(numbers)
    unpacked, _ = unpack_gammas(packed)
    assert unpacked == numbers
