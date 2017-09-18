"""
    gf2m.py: A Python library for operations over a prime field GF(p)
    Author: Jun Kurihara <kurihara at ieee.org>
"""

PRIME_LIST = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
              107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
              227, 229, 233, 239, 241, 251, 257]


class GFp:
    MAX_PRIME = max(PRIME_LIST)

    # constructor
    def __init__(self):
        self.prime = 0x00

    def set_prime(self, size: int) -> None:
        if size > GFp.MAX_PRIME:
            raise Exception("The specified number exceeds the limit. "
                            "It must be less than or equal to {0}.".format(GFp.MAX_PRIME))
        else:
            self.prime = min(filter((lambda x: x >= size), PRIME_LIST))

    '''
    Unlike GF(2^m), we do not need to prepare look up tables for rapid computation.
    This is because every operation over GF(p) is just an arithmetic operation with mod p.
    But we need to consider additive inverse in addition to multiplicative inverse in the case of GF(p).
    '''

    # x + y
    def add_gfp(self, x: int, y: int) -> int:
        assert x in range(self.prime) and y in range(self.prime)
        return (x + y) % self.prime

    # x - y
    def sub_gfp(self, x: int, y: int) -> int:
        assert x in range(self.prime) and y in range(self.prime)
        return self.add_gfp(x, self.add_inv_gfp(y))

    # x + y
    def mul_gfp(self, x: int, y: int) -> int:
        assert x in range(self.prime) and y in range(self.prime)
        return (x * y) % self.prime

    # x / y
    def div_gfp(self, x: int, y: int) -> int:
        assert x in range(self.prime), y in range(self.prime)
        return self.mul_gfp(x, self.mul_inv_gfp(y))

    # -x
    def add_inv_gfp(self, x: int) -> int:
        assert x in range(self.prime) and x != 0
        return self.prime - x

    # 1/x
    def mul_inv_gfp(self, x: int) -> int:
        assert x in range(self.prime) and x != 0
        p = x
        while self.mul_gfp(p, x) != 1:
            p = self.mul_gfp(p, x)
        return p
