"""
    file_wrapper.py: A Python wrapper to handle byte objects
    Author: Jun Kurihara <kurihara at ieee.org>
"""
import numpy as np

DATA_TYPE = np.uint8
FILE_NAME = '../SampleData/rashomon.txt'


def get_npbytes_from_file(infile_name: str) -> np.ndarray:
    file = open(infile_name, "rb")
    indata = file.read()
    file.close()
    return npbytes_from_buffer(indata)


def npbytes_from_buffer(inbuffer: bytes) -> np.ndarray:
    # todo: assertion
    return np.frombuffer(inbuffer, DATA_TYPE)


def npbytes_to_buffer(npdata: np.ndarray) -> bytes:
    # todo: assertion
    return npdata.tobytes()


if __name__ == '__main__':
    np.set_printoptions(formatter={"int": hex})

    numpydata = get_npbytes_from_file(FILE_NAME)
    print(type(numpydata))
    print(numpydata)

    buf = npbytes_to_buffer(numpydata)
    print(type(buf))
    print(buf)
