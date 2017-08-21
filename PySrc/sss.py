# sss.py
# Author: Jun Kurihara <kurihara@ieee.org>

from PySrc.gf2m import GF2m
import numpy as np
from typing import List


class SSS:
    # constructor
    def __init__(self):
        self.__secret = None  # private
        self.__orig_secret_size = 0  # private
        self.__random = None  # private
        self.thre = 0
        self.ramp = 0
        self.num = 0
        self.gf = GF2m()
        self.coefficient_matrix = None
        self.shares = None
        self.share_index_list = None
        # set field is needed

    # private functions
    def __generate_coefficient_matrix(self) -> None:
        self.coefficient_matrix = np.ones(num, np.int)
        self.coefficient_matrix = np.vstack((self.coefficient_matrix, np.arange(1, num + 1, 1)))
        for i in range(self.thre - 2):
            self.coefficient_matrix = np.vstack((self.coefficient_matrix, self.gf.vmul_gf2m(self.coefficient_matrix[1], self.coefficient_matrix[i + 1])))

        self.coefficient_matrix = self.coefficient_matrix.transpose()
        # print("{0}: coefficient (Vandermonde) matrix = \n{1}".format(self, self.coef))

    def __generate_random(self) -> None:
        self.__random = np.random.randint(0, self.gf.mord, [self.thre - self.ramp, int(self.__secret.size / self.ramp)])
        # print("{0}: Set random values = {1}".format(self, self.__random))
        # todo: this method is tentatively implemented! this must be modified by using external crypto libraries.

    # public functions
    def initialize(self, deg, thre, ramp, num) -> None:
        self.__init__()
        self.gf.set_degree(deg)
        self.set_params(thre, ramp, num)
        self.__generate_coefficient_matrix()

    def set_params(self, thre, ramp, num):
        #print("{0}: Set params: (k,l,n) = ({1}, {2}, {3})".format(self, thre, ramp, num))
        self.thre = thre
        self.ramp = ramp
        self.num = num

    def generate_shares(self) -> None:
        self.__generate_random()
        concat = np.vstack((self.__secret, self.__random)).transpose()
        self.shares = []
        self.share_index_list = []
        for i in range(self.num):
            tmp = self.gf.vmul_gf2m(concat, self.coefficient_matrix[i]).transpose()
            self.shares.append(tmp[0])
            for j in range(self.thre - 1):
                self.shares[i] = self.gf.vadd_gf2m(self.shares[i], tmp[j + 1])
            self.share_index_list.append(i)
            #print("{0}: share {1} = {2}".format(self, i, self.shares[i]))

    def set_secret(self, secret: np.ndarray) -> None:
        self.__orig_secret_size = secret.size
        if secret.size % ramp != 0:
            app = np.array([0] * (ramp - secret.size % ramp))
            secret = np.append(secret, app)
        self.__secret = np.resize(secret, [ramp, int(secret.size / ramp)])
        #print("{0}: Set a secret = {1}".format(self, self.__secret))

    def set_external_shares(self, index_list: List) -> None:
        pass

    def reconstruct_secret(self) -> None:
        pass


if __name__ == '__main__':
    s = SSS()
    deg = 8
    thre = 3
    ramp = 1
    num = 4
    parallel = 10

    np.set_printoptions(formatter={'int': hex})

    print("Params: (k, l, n, degree) = ({0}, {1}, {2}, {3})".format(thre, ramp, num, deg))
    nvec = np.random.randint(0, (1 << deg) - 1, parallel)
    print("Secret: {0}".format(nvec))
    s.initialize(deg, thre, ramp, num)
    s.set_secret(nvec)
    s.generate_shares()

    for i in range(num):
        print("Share {0}: {1}".format(s.share_index_list[i], s.shares[i].astype(np.uint8)))
