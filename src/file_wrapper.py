"""
    file_wrapper.py: A Python wrapper to handle byte objects/files
    Author: Jun Kurihara <kurihara at ieee.org>
"""
from ctypes import *

import numpy as np

DATA_TYPE = np.uint8
FILE_NAME = '../SampleData/rashomon.txt'

HEADER_TYPE = 0xFFFFFF01
HEADER_LEN = 16


class share_header(BigEndianStructure):  # Network byte order
    _fields_ = (
        ('type', c_uint32),  # indicates a share
        # value field
        # todo: params of sss needed to be included for self-contained implementation
        ('orig_len', c_uint32),  # original length of the secret
        ('payload_len', c_uint32),  # origitnal length of the share
        ('index', c_int32),  # index
    )


def get_npbytes_from_file(infile_name: str) -> np.ndarray:
    file = open(infile_name, "rb")
    indata = file.read()
    file.close()
    return npbytes_from_buffer(indata)


def get_sharefile_from_npbytes(file_name: str, orig_len: c_uint32, index: c_uint32, nppayload: np.ndarray) -> None:
    # todo: set padding tailor for alignment
    header = create_header(orig_len, len(nppayload), index)
    outfile = open(file_name, "wb")
    outfile.write(memoryview(header).tobytes() + npbytes_to_buffer(nppayload))
    outfile.close()


def get_share_npbytes_from_file(share_file_name: str) -> [c_uint32, c_uint32, c_uint32, np.ndarray]:
    file = open(share_file_name, "rb")
    indata = file.read()
    file.close()
    # todo: truncate padding tailor for alignment
    return [hex(int.from_bytes(indata[share_header.type.offset:share_header.orig_len.offset], "big")),
            int.from_bytes(indata[share_header.orig_len.offset:share_header.payload_len.offset], "big"),
            int.from_bytes(indata[share_header.payload_len.offset:share_header.index.offset], "big"),
            npbytes_from_buffer(indata[HEADER_LEN:])]


def npbytes_from_buffer(inbuffer: bytes) -> np.ndarray:
    # todo: assertion
    return np.frombuffer(inbuffer, DATA_TYPE)


def npbytes_to_buffer(npdata: np.ndarray) -> bytes:
    # todo: assertion
    # change data type to np.uint8
    return npdata.astype(np.uint8).tobytes()


def create_header(orig_len: c_uint32, payload_len: c_uint32, index: c_uint32) -> bytes:
    # todo:assertion
    return share_header(HEADER_TYPE, orig_len, payload_len, index)


if __name__ == '__main__':
    np.set_printoptions(formatter={"int": hex})

    numpydata = get_npbytes_from_file(FILE_NAME)
    print(numpydata)
    print(numpydata.size)

    get_sharefile_from_npbytes("../SampleData/test.txt", numpydata.size, 0, numpydata)
    xyz = get_share_npbytes_from_file("../SampleData/test.txt")

    print(xyz[:-1])
    print(np.allclose(numpydata, xyz[-1:]))
