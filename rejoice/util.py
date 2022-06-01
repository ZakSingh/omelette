class BytesIntEncoder:

    @staticmethod
    def encode(b: bytes) -> int:
        return int.from_bytes(b, byteorder='big')

    @staticmethod
    def decode(i: int) -> bytes:
        return i.to_bytes(((i.bit_length() + 7) // 8), byteorder='big')
