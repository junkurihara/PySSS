"""
    sample.py: A sample python code to do secret sharing using sss.py
    Author: Jun Kurihara <kurihara at ieee.org>
"""

import sss
import numpy as np
import file_wrapper

FILE_NAME = '../SampleData/rashomon.txt'
DEGREE = 8


def main():
    # set params
    threshold = 5
    ramp = 3
    num = 10
    index_list = [1, 2, 3, 4, 5]
    shares = []

    # read file to numpy array
    orig_secret = file_wrapper.get_npbytes_from_file(FILE_NAME)
    orig_size = orig_secret.size

    np.set_printoptions(formatter={'int': hex})

    # share generation
    s = sss.SSS()
    print("Share generation:")
    print("Params: (k, l, n) = ({0}, {1}, {2}) over GF(2^{3})".format(threshold, ramp, num, DEGREE))
    s.initialize(DEGREE, threshold, ramp, num)  # initialize with given parameters
    print("Secret: {0}".format(orig_secret))
    print("Secret size: {0} bytes".format(int(orig_size*DEGREE/8)))
    s.set_secret(orig_secret)  # set the secret into the instance
    s.generate_shares()  # execute share generation with given parameters and secret
    for i in range(num):
        print("Share {0}: {1}".format(i, s.get_shares()[i]))

    # todo: define share file format
    # todo: function -> write share bytes to file
    # todo: function -> parse share file format for recovery

    for i in index_list:
        shares.append(s.get_shares()[i])
    # secret reconstruct
    s.__init__()  # just remove all instance variables
    print("\nSecret reconstruction:")
    s.initialize(DEGREE, threshold, ramp, num)  # initialize with given parameters
    s.set_external_shares(shares, index_list)  # set given shares in the instance
    print("Given params: (k, l, n, share indices, original size) = ({0}, {1}, {2}, {3}, {4}) over GF(2^{5})"
          .format(threshold, ramp, num, index_list, orig_size, DEGREE))
    for i, j in zip(index_list, shares):
        print("Given share {0}: {1}".format(i, j))
    s.reconstruct_secret(orig_size)  # execute secret reconstruction with given shares under given parameter setting
    reco_secret = s.get_secret()
    print("Reconstructed secret: {0}".format(reco_secret))
    # check if the original secret coincides with the reconstructed one
    print("Original secret == Reconstructed secret ?: {0}".format(np.allclose(orig_secret, reco_secret)))

if __name__ == '__main__':
    main()
