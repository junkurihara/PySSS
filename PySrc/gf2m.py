# gf2m.py: A Python library for GF(2^m) operations
# Author: Jun Kurihara <kurihara at ieee.org>

# import sys
import numpy as np
from typing import Sequence, TypeVar, List

DEGREELIST = [2, 4, 8, 10, 12, 16]
POLYDIC = {2: 0x03, 4: 0x03, 8: 0x1d, 10: 0x09, 12: 0x53, 16: 0x010b}  # Degree:Poly x^12 + x^6 + x^4 + x^1 + 1


class GF2m:
    MAXDEGREE = max(DEGREELIST)
    T = TypeVar('T')

    # constructor
    def __init__(self):
        self.deg = 0x00  # degree of extension
        self.mord = 0x00  # multiplicative order
        self.poly = 0x00  # irreducible polynomial
        self.mul = None  # table of multiplicative cyclic group
        self.idx = None  # index table for reference to mul[]

    def set_degree(self, size: int) -> None:
        if size > GF2m.MAXDEGREE:
            raise Exception("The specified degree m exceeds the limit. It must be less than or equal to {0}.".format(
                GF2m.MAXDEGREE))
        else:
            self.deg = min(filter((lambda x: x >= size), DEGREELIST))
            self.mord = (1 << self.deg) - 1  # multiplicative order
            # print("{0}: (deg, order, mult order) = ({1}, {2}, {3})".format(self, self.deg, self.mord+1, self.mord))
            try:
                self.poly = POLYDIC[self.deg]
                # print("{0}: Polynomial = {1}".format(self, hex(self.poly+(1<<self.deg))))
            except KeyError:
                print("{0}: Polynomial of degree {1} is not defined".format(self, self.deg))
            self.mul = self.__gen_mul_table(self.mord, self.poly)
            self.idx = self.__gen_idx_table(self.mord, self.mul)

    # private functions
    def __gen_mul_table(self, mord: int, poly: int) -> List[int]:
        # print("{0}: Generate multiplicative table mul[] of the primitive element".format(self))
        poly ^= (mord + 1)
        mul = [0x01]
        for i in range(mord - 1):  # multiplicative order is 2^m-1, |F^*(2^m)|=2^m-1
            p = mul[i] << 1
            mul.append(p ^ poly if p > mord else p)
        return mul

    def __gen_idx_table(self, mord: int, mul: Sequence[T]) -> List[int]:
        # print("{0}: Generate index table idx[] that maps the value to the exponent
        #  (e.g., when mul[i] = j, idx[j] = i)".format(self))
        idx = [-1] * (mord + 1)  # 0 is not in multiplicative table
        for i in range(mord + 1):
            for j in range(mord):
                if mul[j] == i:
                    idx[i] = j
        return idx

    # public functions
    # vectorized addition over GF(2^m) for ndarray
    def vadd_gf2m(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda a, b: a ^ b)(x, y)

    # vectorized multiplication over GF(2^m) for ndarray
    def vmul_gf2m(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda a, b: self.mul[(self.idx[a] + self.idx[b]) % self.mord])(x, y)

    # vectorized division over GF(2^m) for ndarray
    def vdiv_gf2m(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda a, b: self.mul[((self.mord + (self.idx[a] - self.idx[b])) % self.mord)])(x, y)

    # vectorized inversion over GF(2^m) for ndarray
    def vinv_gf2m(self, x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda a: self.mul[((self.mord - self.idx[a]) % self.mord)])(x)

    # addition over GF(2^m)
    def add_gf2m(self, x: int, y: int) -> int:
        if x > self.mord or y > self.mord or x < 0 or x < 0:
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))
        return x ^ y

    # multiplication over GF(2^m)
    def mul_gf2m(self, x: int, y: int) -> int:
        try:
            i = (self.idx[x] + self.idx[y]) % self.mord
            return self.mul[i]
        except IndexError:
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))

    # division over GF(2^m)
    def div_gf2m(self, x: int, y: int) -> int:
        try:
            i = ((self.mord + (self.idx[x] - self.idx[y])) % self.mord)
            return self.mul[i]
        except IndexError:
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))

    # inversion over GF(2^m)
    def inv_gf2m(self, x: int) -> int:
        if (x > self.mord or x < 0):
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))
        return self.mul[((self.mord - self.idx[x]) % self.mord)]


if __name__ == '__main__':
    deg = 8
    vsize = 4  # can be matrix
    g = GF2m()
    g.set_degree(deg)
    nvec1 = np.random.randint(0, g.mord, vsize)
    nvec2 = np.random.randint(0, g.mord, vsize)
    print("multiplicative table of GF(2^{0}) = {1}".format(deg, g.mul))
    print("index table of GF(2^{0}) = {1}".format(deg, g.idx))
    print("nvec1 = {0}".format(nvec1))
    print("nvec2 = {0}".format(nvec2))
    print("nvec1 + nvec2 = {0}".format(g.vadd_gf2m(nvec1, nvec2)))
    print("nvec1 * nvec2 = {0}".format(g.vmul_gf2m(nvec1, nvec2)))
    print("nvec1 / nvec2 = {0}".format(g.vdiv_gf2m(nvec1, nvec2)))
    print("nvec1^-1 = {0}".format(g.vinv_gf2m(nvec1)))
