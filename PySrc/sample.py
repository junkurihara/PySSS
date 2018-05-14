"""
    sample.py: A sample python code to do secret sharing using sss.py
    Author: Jun Kurihara <kurihara at ieee.org>
"""

import rs_sss
import numpy as np
import file_wrapper

FILE_NAME = '../SampleData/rashomon.txt'
FILE_NAME_SHARE_EXT = '.sss'
DEGREE = 8


def main():
    # set params
    threshold = 3
    ramp = 2
    num = 5
    index_list = [1, 2, 4]
    shares = []

    # assertion to check if given parameters is appropriate
    assert 1 < ramp < threshold
    assert threshold < num
    assert len(index_list) >= threshold

    # read file to numpy array
    orig_secret = file_wrapper.get_npbytes_from_file(FILE_NAME)
    orig_size = orig_secret.size

    np.set_printoptions(formatter={'int': hex})

    get_share_name = (lambda idx: FILE_NAME + str(idx) + FILE_NAME_SHARE_EXT)

    # share generation
    s = rs_sss.RS_SSS(DEGREE)
    print("Share generation:")
    print("Params: (k, l, n) = ({0}, {1}, {2}) over GF(2^{3})".format(threshold, ramp, num, DEGREE))
    s.initialize(threshold, ramp, num)  # initialize with given parameters
    print("Secret file: {0}".format(FILE_NAME))
    print("Secret: {0}".format(orig_secret))
    print("Secret size: {0} bytes".format(int(orig_size * DEGREE / 8)))
    s.set_secret(orig_secret)  # set the secret into the instance
    s.generate_shares()  # execute share generation with given parameters and secret
    for i in range(num):
        print("Share {0}, file = {1}: {2}".format(i, get_share_name(i), s.get_shares()[i]))
        # write shares to files
        file_wrapper.get_sharefile_from_npbytes(get_share_name(i), orig_size, i, s.get_shares()[i])

    for i in index_list:
        # shares.append(s.get_shares()[i])
        shares.append(file_wrapper.get_share_npbytes_from_file(get_share_name(i))[-1:])

    # secret reconstruct
    s.__init__(DEGREE)  # just remove all instance variables
    print("\nSecret reconstruction:")
    s.initialize(threshold, ramp, num)  # initialize with given parameters
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
