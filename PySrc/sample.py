"""
    sample.py: A sample python code to do secret sharing using sss.py
    Author: Jun Kurihara <kurihara at ieee.org>
"""

import sss
import numpy as np


def main():
    deg = 8
    threshold = 8
    ramp = 3
    num = 11
    orig_size = 1024
    index_list = [0, 1, 2, 8, 4, 7, 9, 10]
    shares = []

    np.set_printoptions(formatter={'int': hex})

    # share generation
    s = sss.SSS()
    print("Share generation:")
    print("Params: (k, l, n) = ({0}, {1}, {2}) over GF(2^{3})".format(threshold, ramp, num, deg))
    s.initialize(deg, threshold, ramp, num)  # initialize with given parameters
    orig_secret = np.random.randint(0, (1 << deg) - 1, orig_size)
    print("Secret: {0}".format(orig_secret))
    s.set_secret(orig_secret)  # set the secret into the instance
    s.generate_shares()  # execute share generation with given parameters and secret
    for i in range(num):
        print("Share {0}: {1}".format(i, s._shares[i]))

    for i in index_list:
        shares.append(s._shares[i])
    # secret reconstruct
    s.__init__()  # just remove all instance variables
    print("\nSecret reconstruction:")
    s.initialize(deg, threshold, ramp, num)  # initialize with given parameters
    s.set_external_shares(shares, index_list)  # set given shares in the instance
    print("Given params: (k, l, n, share indices, original size) = ({0}, {1}, {2}, {3}, {4}) over GF(2^{5})"
          .format(threshold, ramp, num, s._share_index_list, org_size, deg))
    for i, j in zip(s._share_index_list, s._shares):
        print("Given share {0}: {1}".format(i, j))
    s.reconstruct_secret(orig_size)  # execute secret reconstruction with given shares under given parameter setting
    print("Reconstructed secret: {0}".format(s._secret))
    # check if the original secret coincides with the reconstructed one
    print("Original secret == Reconstructed secret ?: {0}".format(np.allclose(orig_secret, s._secret)))

if __name__ == '__main__':
    main()
