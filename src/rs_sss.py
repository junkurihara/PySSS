"""
    rs_sss.py: A Python library for secret sharing scheme based on polynomial interpolation
    (Reed-Solomon code-based secret sharing scheme)
    Author: Jun Kurihara <kurihara at ieee.org>
"""

import gf2m
import numpy as np
from sss import SSS

DATA_TYPE = np.uint16


class RS_SSS(SSS):
    # constructor
    def __init__(self, degree: int):
        super().__init__()
        self.gf = gf2m.GF2m()
        self.gf.set_degree(degree)
        self.coefficient_matrix = None

    # private functions
    def __set_coefficient_matrix(self) -> None:
        # todo: Reconsider how to set indices
        # this method is implemented under an assumption that indices are enumerated
        # not as multiplicative elements in GF(2^m) [1, a^1, ..., a^{n-1}] but just as integers [1, 2, ..., n]
        self.coefficient_matrix = np.ones(self._num, DATA_TYPE)
        self.coefficient_matrix = np.vstack((self.coefficient_matrix, np.arange(1, self._num + 1, 1)))
        for i in range(self._threshold - 2):
            self.coefficient_matrix = np.vstack((self.coefficient_matrix,
                                                 self.gf.vmul_gf2m(
                                                     self.coefficient_matrix[1],
                                                     self.coefficient_matrix[i + 1])))

        self.coefficient_matrix = self.coefficient_matrix.transpose()

    # public functions
    def initialize(self, threshold: int, ramp: int, num: int) -> None:
        self._set_params(threshold, ramp, num)
        self.__set_coefficient_matrix()

    def generate_shares(self) -> None:
        self._generate_random(self.gf.mord, DATA_TYPE)
        # todo: reconsider this algorithm of waste of memory
        concat = np.vstack((self._secret, self.random)).transpose()
        self._shares = []
        self._share_index_list = []
        # todo: define individual method in gf2m.py
        for i in range(self._num):
            tmp = self.gf.vmul_gf2m(concat, self.coefficient_matrix[i]).transpose()
            self._shares.append(tmp[0])
            for j in range(self._threshold - 1):
                self._shares[i] = self.gf.vadd_gf2m(self._shares[i], tmp[j + 1])
            self._share_index_list.append(i)

    def set_secret(self, secret: np.ndarray) -> None:
        self.orig_secret_size = secret.size
        if secret.size % self._ramp != 0:
            app = np.array([0] * (self._ramp - (secret.size % self._ramp)), DATA_TYPE)
            secret = np.append(secret, app)
        self._secret = np.resize(secret, [self._ramp, int(secret.size / self._ramp)])

    def reconstruct_secret(self, orig_size: int) -> None:
        assert len(self._shares) == self._threshold, "# of given shares is not the threshold k"

        self.orig_secret_size = orig_size
        submatrix = self.coefficient_matrix[self._share_index_list[0]]
        for i in self._share_index_list[1:]:
            submatrix = np.vstack((submatrix, self.coefficient_matrix[i]))

        submatrix = self.gf.inverse_matrix_gf2m(submatrix.transpose()).transpose()
        self._shares = np.array(self._shares).transpose()

        sec = np.zeros((self._ramp, self._shares.shape[0]), dtype=DATA_TYPE)
        for i in range(self._ramp):
            tmp = self.gf.vmul_gf2m(submatrix[i], self._shares).transpose()
            for j in range(self._threshold):
                sec[i] = self.gf.vadd_gf2m(sec[i], tmp[j])

        # reshape the secret matrix and set as an instance variable
        self._secret = np.reshape(sec, (1, -1))[0][:self.orig_secret_size]
