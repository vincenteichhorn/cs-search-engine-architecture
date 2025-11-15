from libc.math cimport ceil

cdef class BitWriter:
    cdef bytearray buffer
    cdef unsigned char current_byte
    cdef int bits_filled
    cdef public long total_bits_written

    def __cinit__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_filled = 0
        self.total_bits_written = 0

    cpdef void write_bits(self, unsigned int value, int bit_count):
        """
        Writes the lowest `bit_count` bits of `value` to the buffer.
        Bits are written MSB-first.
        """
        cdef int i
        cdef unsigned int bit

        for i in range(bit_count - 1, -1, -1):
            bit = (value >> i) & 1
            self.current_byte = (self.current_byte << 1) | bit
            self.bits_filled += 1
            self.total_bits_written += 1

            if self.bits_filled == 8:
                self.buffer.append(self.current_byte)
                self.current_byte = 0
                self.bits_filled = 0

    cpdef void flush(self):
        """
        Writes any remaining bits to the buffer, padding the rest of the byte with zeros.
        """
        if self.bits_filled > 0:
            # shift the remaining bits to the MSB of the byte
            self.current_byte <<= (8 - self.bits_filled)
            self.buffer.append(self.current_byte & 0xFF)
            self.current_byte = 0
            self.bits_filled = 0

    cpdef bytes get_bytes(self):
        """
        Flushes and returns the contents of the buffer as immutable bytes.
        """
        self.flush()
        return bytes(self.buffer)


cdef class BitReader:
    cdef bytes data
    cdef Py_ssize_t byte_index
    cdef int bit_index

    def __cinit__(self, bytes data):
        self.data = data
        self.byte_index = 0
        self.bit_index = 0

    cpdef int read_bits(self, int bit_count):
        cdef int value = 0
        cdef unsigned char current_byte, bit
        cdef Py_ssize_t data_len = len(self.data)

        for _ in range(bit_count):
            if self.byte_index >= data_len:
                raise EOFError("No more bits to read")

            current_byte = self.data[self.byte_index]
            bit = (current_byte >> (7 - self.bit_index)) & 1
            value = (value << 1) | bit

            self.bit_index += 1
            if self.bit_index == 8:
                self.bit_index = 0
                self.byte_index += 1
        return value


    cpdef bytes bytes_remaining(self):
        cdef Py_ssize_t start
        if self.bit_index == 0:
            start = self.byte_index
        else:
            start = self.byte_index + 1
        return self.data[start:]
    
    cpdef int bits_remaining(self):
        cdef Py_ssize_t data_len = len(self.data)
        return (data_len - self.byte_index) * 8 - self.bit_index - 1


cpdef int encode_gamma(BitWriter writer, int number):
    """Encodes a positive integer using gamma coding and writes it to the BitWriter.
    Gamma coding represents a positive integer n as:
    - The unary representation of floor(log2(n)) (i.e., leading zeros followed by a one)
    - Followed by the binary representation of n without its leading 1 bit.

    Example:
        For n = 10 (binary 1010):
        - floor(log2(10)) = 3, unary = 0001
        - binary without leading 1 = 010
        - gamma code = 0001 010
        
    Args:
        writer (BitWriter): The BitWriter to write the encoded bits to.
        number (int): The positive integer to encode.
    Returns:
        int: The number of bits written.
    """
    if number <= 0:
        raise ValueError(f"Gamma code is only for positive integers, got {number}")

    cdef int bitlen = number.bit_length() - 1
    writer.write_bits(0, bitlen)
    writer.write_bits(number, bitlen + 1)
    return bitlen * 2 + 1


cpdef tuple decode_gamma(BitReader reader):
    """Decodes a gamma-coded integer from the BitReader.

    Args:
        reader (BitReader): The BitReader to read the encoded bits from.
    Returns:
        tuple: A tuple containing the decoded integer and the number of bits read.
    """
    cdef int leading_zeros = 0
    cdef int bit, number, bits_used

    while reader.read_bits(1) == 0:
        leading_zeros += 1

    number = 1
    for _ in range(leading_zeros):
        bit = reader.read_bits(1)
        number = (number << 1) | bit

    bits_used = leading_zeros + 1 + leading_zeros
    return number, bits_used


cpdef bytes pack_gammas(object writer, list numbers):
    """Packs a list of positive integers into a bytes object using gamma coding.
    The nubers are packend as follows:
    - First, the total number of bits used to encode all integers is gamma-encoded.
    - Then, each integer is gamma-encoded and concatenated.

    Example:
        For numbers = [3, 5]:
        - 3 is encoded as 0011
        - 5 is encoded as 000101
        - Total bits = 4 + 6 = 10, which is encoded as 000001010
        - Final bit sequence = [total bits encoding] + [3 encoding] + [5 encoding]
        - Converted/padded to bytes.

    Args:
        numbers (list): A list of positive integers to encode.
    Returns:
        bytes: A bytes object containing the gamma-coded integers.
    """
    cdef BitWriter inner_writer = BitWriter()
    cdef int n
    for n in numbers:
        encode_gamma(inner_writer, n)

    cdef long total_bits = inner_writer.total_bits_written
    encode_gamma(writer, total_bits)

    for b in inner_writer.get_bytes():
        writer.write_bits(b, 8)


cpdef list unpack_gammas(object reader, int read_n=-1):
    """Unpacks a bytes object containing gamma-coded integers into a list of integers.
    The bytes are unpacked as follows:
    - First, the total number of bits used to encode all integers is gamma-decoded.
    - Then, integers are gamma-decoded until the total number of bits is reached.

    Example:
        bytes = gamma-coded representation of [3, 5], thus containing:
        - Total bits encoding = 000001010 (10 bits)
        - 3 encoding = 0011
        - 5 encoding = 000101
        Decoding reads total bits first, then decodes integers until 10 bits are read.
        Remaining bits until byte boundary are ignored. Remaining bytes from input are returned.

    Args:
        data (bytes): A bytes object containing the gamma-coded integers.
    Returns:
        tuple: A tuple containing the list of decoded integers and any remaining bytes from the input which were not used.
    """

    cdef int total_bits
    cdef int bits_read = 0
    cdef list numbers = []
    bits_needed, _ = decode_gamma(reader)

    print(f"Decoding {bits_needed} bits using gamma coding")
    cdef int no_ceil_bits = 8 - (bits_needed % 8)

    while bits_read < bits_needed and (read_n == -1 or len(numbers) < read_n):
        n, used = decode_gamma(reader)
        bits_read += used
        numbers.append(n)
        
    reader.read_bits(no_ceil_bits) 
    print(f"skipped {no_ceil_bits} bits to align to byte boundary")
    return numbers
