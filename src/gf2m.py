"""
    gf2m.py: A Python library for GF(2^m) operations
    Author: Jun Kurihara <kurihara at ieee.org>
"""

import numpy as np
from typing import Sequence, TypeVar, List

DEGREE_LIST = [2, 4, 8, 10, 12, 16]
POLYNOMIAL_DIC = {2: 0x03, 4: 0x03, 8: 0x1d, 10: 0x09, 12: 0x53, 16: 0x010b}  # Degree:Poly x^12 + x^6 + x^4 + x^1 + 1
DATA_TYPE = np.uint16


class GF2m:
    MAX_DEGREE = max(DEGREE_LIST)
    T = TypeVar('T')

    # constructor
    def __init__(self):
        self.deg = 0x00  # degree of extension
        self.mord = 0x00  # multiplicative order
        self.poly = 0x00  # irreducible polynomial
        self.mul = None  # table of multiplicative cyclic group
        self.idx = None  # index table for reference to mul[]

    def set_degree(self, size: int) -> None:
        if size > GF2m.MAX_DEGREE:
            raise Exception("The specified degree m exceeds the limit. It must be less than or equal to {0}.".format(
                GF2m.MAX_DEGREE))
        else:
            self.deg = min(filter((lambda x: x >= size), DEGREE_LIST))
            self.mord = (1 << self.deg) - 1  # multiplicative order
            # print("{0}: (deg, order, mult order) = ({1}, {2}, {3})".format(self, self.deg, self.mord+1, self.mord))
            try:
                self.poly = POLYNOMIAL_DIC[self.deg]
                # print("{0}: Polynomial = {1}".format(self, hex(self.poly+(1<<self.deg))))
            except KeyError:
                print("{0}: Polynomial of degree {1} is not defined".format(self, self.deg))
            self.mul = self.__gen_mul_table(self.mord, self.deg, self.poly)
            self.idx = self.__gen_idx_table(self.mord, self.mul)

    # private functions
    @staticmethod
    def __gen_mul_table(mord: int, deg: int, poly: int) -> List[int]:
        # print("{0}: Generate multiplicative table mul[] of the primitive element".format(self))
        threshold = 1 << (deg - 1)  # to avoid buffer overflow just in case
        mul = [0x01]
        for i in range(mord - 1):  # multiplicative order is 2^m-1, |F^*(2^m)|=2^m-1
            if mul[i] < threshold:
                p = mul[i] << 1
            else:
                p = ((mul[i] ^ threshold) << 1) ^ poly
            mul.append(p)
        return mul

    @staticmethod
    def __gen_idx_table(mord: int, mul: Sequence[T]) -> List[int]:
        # print("{0}: Generate index table idx[] that maps the value to the exponent
        #  (e.g., when mul[i] = j, idx[j] = i)".format(self))
        idx = [-1] * (mord + 1)  # 0 is not in multiplicative table
        for i in range(mord + 1):
            for j in range(mord):
                if mul[j] == i:
                    idx[i] = j
        return idx

    # public functions

    # Gauss-Jordan Elimination to obtain inverse matrix over GF(2^m)
    # this code is implemented under the assumption that the input matrix is non-singular
    def inverse_matrix_gf2m(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.concatenate((matrix, np.identity(matrix.shape[0], dtype=DATA_TYPE)), axis=1)
        for i in range(matrix.shape[0]):
            # pivoting
            k = i
            while matrix[k][i] == 0 and k < matrix.shape[0] - 1:
                if matrix[k + 1][i] != 0:
                    tmp_m = np.copy(matrix[i])
                    matrix[i] = matrix[k + 1]
                    matrix[k + 1] = tmp_m
                    break
                else:
                    k += 1
            # forward and backward elimination
            matrix[i] = self.vmul_gf2m(self.vinv_gf2m(matrix[i][i]), matrix[i])
            for j in range(matrix.shape[0]):
                if j != i:
                    matrix[j] ^= self.vmul_gf2m(matrix[j][i], matrix[i])
                    # print(matrix)

        return matrix[:, matrix.shape[0]:]

    # followings are vectorized implementations
    # vectorized addition over GF(2^m) for ndarray
    @staticmethod
    def vadd_gf2m(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda a, b: a ^ b)(x, y).astype(DATA_TYPE)

    # vectorized multiplication over GF(2^m) for ndarray
    def vmul_gf2m(self, x: object, y: object) -> np.ndarray:
        return np.vectorize(lambda a, b:
                            self.mul[(self.idx[a] + self.idx[b]) % self.mord]
                            if a != 0 and b != 0 else 0)(x, y).astype(DATA_TYPE)

    # vectorized division over GF(2^m) for ndarray
    def vdiv_gf2m(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda a, b:
                            self.mul[((self.mord + (self.idx[a] - self.idx[b])) % self.mord)]
                            if a != 0 else 0)(x, y).astype(DATA_TYPE)

    # vectorized inversion over GF(2^m) for ndarray
    def vinv_gf2m(self, x: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda a:
                            self.mul[((self.mord - self.idx[a]) % self.mord)]
                            if a != 0 else 0)(x).astype(DATA_TYPE)

    # followings are naive implementations
    # addition over GF(2^m)
    def add_gf2m(self, x: int, y: int) -> int:
        if x > self.mord or y > self.mord or x < 0 or x < 0:
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))
        return x ^ y

    # multiplication over GF(2^m)
    def mul_gf2m(self, x: int, y: int) -> int:
        try:
            if x == 0 or y == 0:
                return 0
            else:
                i = (self.idx[x] + self.idx[y]) % self.mord
                return self.mul[i]
        except IndexError:
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))

    # division over GF(2^m)
    def div_gf2m(self, x: int, y: int) -> int:
        try:
            if y == 0:
                raise Exception(ValueError)
            else:
                return self.mul_gf2m(x, self.inv_gf2m(y))
        except IndexError:
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))

    # inversion over GF(2^m)
    def inv_gf2m(self, x: int) -> int:
        try:
            if x == 0:
                return 0
            else:
                return self.mul[((self.mord - self.idx[x]) % self.mord)]
        except IndexError:
            print("{0}: Range from 0x00 to {1} must be specified".format(self, hex(self.mord)))


"""
 the following is a test code
"""


def test_gf2m():
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
    print("2 * nvec1 = {0}".format(g.vmul_gf2m(2, nvec1)))

    mat = np.array([[1, 1, 1, 1], [1, 2, 3, 4], [1, 4, 5, 16], [1, 8, 15, 64]], dtype=np.int)

    print(g.vmul_gf2m(np.array([1, 2, 3, 4]), mat))
    print(g.vmul_gf2m(mat[1], mat[2]))
    print("matrix =\n{0}".format(mat))
    print("inverse matrix =\n{0}".format(g.inverse_matrix_gf2m(mat)))


if __name__ == '__main__':
    test_gf2m()
