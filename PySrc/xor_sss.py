"""
    rs_sss.py: A Python library for XOR-based secret sharing scheme
    Author: Jun Kurihara <kurihara at ieee.org>
"""

import numpy as np
from sss import SSS


class RS_SSS(SSS):
    # constructor
    def __init__(self, prime: int):
        super().__init__()

    # private functions
    def __set_coefficient_matrix(self) -> None:
        # todo: should this be in a form of coefficient matrix for XOR based scheme? -> reconsider
        pass

    # public functions
    def initialize(self, threshold: int, ramp: int, num: int) -> None:
        pass

    def generate_shares(self) -> None:
        pass

    def set_secret(self, secret: np.ndarray) -> None:
        pass

    def reconstruct_secret(self, orig_size: int) -> None:
        assert len(self._shares) == self._threshold, "# of given shares is not the threshold k"
        pass
