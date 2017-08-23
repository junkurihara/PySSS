# PySSS: Secret sharing scheme over Python

PySSS is a library to execute secret sharing schemes on Python, which currently supports the scheme based on the polynomial interpolation [[1]](#Shamir1979) over m-degree extension of a binary Galois Field GF(2^m).

The aim of this library is to provide an implementation of secret sharing schemes based on the polynomial interpolation, which can be used as the benchmark of secret sharing scheme.

## Overview
The library currently consists of just two python source files, `PySrc/sss.py` and `ySrc/gf2m.py`.

## Requirements
This library requires:
- `Python 3.6` or above
- `Numpy`

## Usage

### Setup

### Share generation

### Secret reconstruction

## License
Licensed under the MIT license, see `LICENSE` file.

## References
<a name="Shamir1979">[1]</a> A. Shamir, ``How to share a secret,'' Communications of the ACM, vol. 22, no. 11, pp. 612–613, Nov. 1979.
