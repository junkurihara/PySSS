# sss.py
# Author: Jun Kurihara <kurihara at ieee.org>

from PySrc.gf2m import GF2m
import numpy as np
from typing import List


class SSS:
    # constructor
    def __init__(self):
        self._secret = None
        self._shares = None
        self._share_index_list = None
        self._threshold = 0
        self._ramp = 0
        self._num = 0
        self.random = None
        self.gf = GF2m()
        self.coefficient_matrix = None
        self.orig_secret_size = 0

    # private functions
    def __set_params(self, threshold, ramp, num):
        #print("{0}: Set params: (k,l,n) = ({1}, {2}, {3})".format(self, threshold, ramp, num))
        self._threshold = threshold
        self._ramp = ramp
        self._num = num

    def __generate_coefficient_matrix(self) -> None:
        self.coefficient_matrix = np.ones(num, np.int)
        self.coefficient_matrix = np.vstack((self.coefficient_matrix, np.arange(1, num + 1, 1)))
        for i in range(self._threshold - 2):
            self.coefficient_matrix = np.vstack((self.coefficient_matrix, self.gf.vmul_gf2m(self.coefficient_matrix[1], self.coefficient_matrix[i + 1])))

        self.coefficient_matrix = self.coefficient_matrix.transpose()
        # print("{0}: coefficient (Vandermonde) matrix = \n{1}".format(self, self.coef))

    def __generate_random(self) -> None:
        self.random = np.random.randint(0, self.gf.mord, [self._threshold - self._ramp, int(self._secret.size / self._ramp)])
        # print("{0}: Set random values = {1}".format(self, self.__random))
        # todo: this method is tentatively implemented! this must be modified by using external crypto libraries.

    # public functions
    def initialize(self, deg, threshold, ramp, num) -> None:
        self.__init__()
        self.gf.set_degree(deg)
        self.__set_params(threshold, ramp, num)
        self.__generate_coefficient_matrix()

    def generate_shares(self) -> None:
        self.__generate_random()
        concat = np.vstack((self._secret, self.random)).transpose()
        self._shares = []
        self._share_index_list = []
        for i in range(self._num):
            tmp = self.gf.vmul_gf2m(concat, self.coefficient_matrix[i]).transpose()
            self._shares.append(tmp[0])
            for j in range(self._threshold - 1):
                self._shares[i] = self.gf.vadd_gf2m(self._shares[i], tmp[j + 1])
            self._share_index_list.append(i)
            #print("{0}: share {1} = {2}".format(self, i, self.shares[i]))

    def set_secret(self, secret: np.ndarray) -> None:
        self.orig_secret_size = secret.size
        if secret.size % ramp != 0:
            app = np.array([0] * (ramp - secret.size % ramp))
            secret = np.append(secret, app)
        self._secret = np.resize(secret, [ramp, int(secret.size / ramp)])
        #print("{0}: Set a secret = {1}".format(self, self.__secret))


    # todo
    def set_external_shares(self, index_list: List) -> None:
        pass

    # todo
    def reconstruct_secret(self) -> None:
        pass


if __name__ == '__main__':
    s = SSS()
    deg = 8
    threshold = 12
    ramp = 4
    num = 14
    parallel = 100

    np.set_printoptions(formatter={'int': hex})

    print("Params: (k, l, n, degree) = ({0}, {1}, {2}, {3})".format(threshold, ramp, num, deg))
    nvec = np.random.randint(0, (1 << deg) - 1, parallel)
    print("Secret: {0}".format(nvec))
    s.initialize(deg, threshold, ramp, num)
    s.set_secret(nvec)
    s.generate_shares()

    for i in range(num):
        print("Share {0}: {1}".format(s._share_index_list[i], s._shares[i]))