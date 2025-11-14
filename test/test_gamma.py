from sea.util.gamma import (
    BitReader,
    BitWriter,
    encode_gamma,
    decode_gamma,
    pack_gammas,
    unpack_gammas,
)
from sea.posting import Posting


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
    reader = BitReader(packed)
    unpacked = unpack_gammas(reader)
    assert unpacked == numbers


def test_pack_unpack_gammas_postings():

    postings = (Posting(1, [0, 2, 4]), Posting(3, [1, 3]), Posting(5, [0, 1, 2, 3]))
    data = bytearray()
    for posting in postings:
        print(f"Posting(doc_id={posting.doc_id}, positions={posting.positions})")
        byts = posting.serialize()
        data.extend(byts)
    reader = BitReader(bytes(data))
    unpacked_postings = []
    while reader.bits_remaining() > 0:
        print(f"Bits remaining: {reader.bits_remaining()}")
        posting = Posting.deserialize(reader)
        print(f"Posting(doc_id={posting.doc_id}, positions={posting.positions})")
        unpacked_postings.append(posting)
    for p1, p2 in zip(postings, unpacked_postings):
        assert p1.doc_id == p2.doc_id
        assert p1.positions == p2.positions
