# PySSS: Secret sharing scheme on Python

PySSS is a library to execute secret sharing schemes on Python.
PySSS currently supports the ordinary _k_-out-of-_n_ threshold ((_k, n_)-threshold) scheme based on the polynomial interpolation [[1]](#Shamir1979) over an _m_-degree extension of the binary Galois field, i.e., _GF(2^m)_. PySSS also supports its extension called the threshold ramp scheme [[2]](#Yamamoto1985) [[3]](#Blakley1985) with parameter _L_, i.e., (_k, L, n_)-threshold ramp scheme.

The aim of this library is to provide an implementation of secret sharing schemes based on the polynomial interpolation, which can be used as a benchmark of secret sharing schemes.

## Status and road map 
The current status of this project is under development. The polynomial interpolation-based schemes have been implemented, but they are currently just naive implementations. So sophistication in terms of coding is required, e.g., handling irregular parameters. Also, there exists several schemes which are based on other techniques like array codes using only exclusive-or operations. In the future road map, such schemes are needed to get implemented as benchmark software.

## Overview
The library currently consists of some python source files. You see `PySrc/sample.py` is a sample code to use these python source files.

`PySrc/gf2m.py` is a python code for addition, multiplication, division and inversion over an m-degree extension field of _GF(2)_.

`PySrc/rs_sss.py` is a naive implementation of a Reed-Solomon code-based (_k, n_)-threshold scheme and (_k, L, n_)-threshold ramp scheme over _GF(2^m)_ , which supports share generation from given secret and secret reconstruction from given shares.

`PySrc/sss.py` defines a abstract base class of secret sharing scheme objects.

`PySrc/file_wrapper` gives some functions to handle byte objects and files.

`PySrc/gfp.py` and `PySrc/xor_sss.py` are now skeletons of XOR-based secret sharing schemes. 

## Requirements
This library requires:
- `Python 3.6` or above
- `Numpy`

## Usage
You can easily see the usage by referring to the sample code `PySrc/sample.py`

### Setup
Whenever you execute share generation or secret reconstruction, you need some set-ups. 
First you need to instantiate the class `RS_SSS()` in `PySrc/rs_sss.py` with a degree of field extension _m_, as
```python
import rs_sss
deg = 8 # m of GF(2^m)
s = rs_sss.RS_SSS(deg)
```
You also need to initialize the instance via `RS_SSS.initialize` with three parameters: threshold _k_, the ramp parameter _L_, the number of shares _n_, as shown in the following sample code.
```python
threshold = 8 # k
ramp = 3  # L
num = 11  # n
s.initialize(threshold, ramp, num)
```
Here, recall that the (_k, 1, n_)-threshold scheme coincides with the (_k, n_)-threshold scheme.
Hence when you set the ramp parameter _L_ = 1 in the initialization phase, you will execute the standard threshold scheme.
Also note that _k_ must be _k_ <= _n_ and _L_ must be 1 < _L_ < _k_.

### Share generation
After you initialize the instance of `RS_SSS`, you can generate shares for a secret of arbitrary length.
In this implementation, the secret is given in the form of one-dimensional `numpy.ndarray`, i.e., a vector, and each element of the vector must be an element of _GF(2^m)_. Namely, each element is of length _m_-bit. The secret is set to the instance via a method `RS_SSS.set_secret`.
In the following example, the secret is generated at random.
```python
# generate a random secret of 1024 bytes (deg = 8)
orig_size = 1024
orig_secret = np.random.randint(0, (1 << deg) - 1, orig_size) 

# set the secret into the instance
s.set_secret(orig_secret) 
```
Then, you are finally able to generate shares via `RS_SSS.generate_shares`.
```python
# generate shares from the given secret
s.generate_shares()

# shares are generated in SSS._shares[]
for i in range(num):
    print("Share {0}: {1}".format(i, s.get_shares()[i]))
```

### Secret reconstruction
In order to reconstruct the secret, you need to set shares and their indices as instance variables via a method `RS_SSS.set_external_shares`. Here we note that shares themselves are given in the form of a list of one-dimensional `numpy.ndarray` and their indices are given by a list of integers.
```python
shares = []  # list of shares for secret reconstruction
index_list = [0, 1, 2, 8, 4, 7, 9, 10]  # list of share indices for secret reconstruction

# copy from the instance variable
for i in index_list:
    shares.append(s.get_shares()[i])
    
# initialize again and remove all data from the instance
s.__init__(deg)
s.initialize(threshold, ramp, num)

# set copied shares and their indices to the instance
s.set_external_shares(shares, index_list)
```
Then you can reconstruct the secret and obtain the reconstructed secret in `RS_SSS._secret` as follows.
```python
# reconstruct the secret
s.reconstruct_secret(orig_size)
reco_secret = s.get_secret()

# check if the original secret coincides with the reconstructed one
print("Original secret == Reconstructed secret ?: {0}".format(np.allclose(orig_secret, reco_secret)))
```
Here we should note that the parameter `orig_size` must be specified. This is because some zeros might have got padded to the given secret in the phase of share generation since the length of the secret must be a multiple of _l_. Hence such padding objects must be removed in the secret reconstruction phase by giving the original length of the secret.

## License
Licensed under the MIT license, see `LICENSE` file.

## References
<a name="Shamir1979">[1]</a> A. Shamir, ``How to share a secret,'' Communications of the ACM, vol. 22, no. 11, pp. 612--613, Nov. 1979.

<a name="Yamamoto1985">[2]</a> H. Yamamoto, ``On secret sharing systems using (_k_, _L_, _n_)-threshold scheme,'' IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences (Japanese Ed.), vol. J68-A, no. 9, pp. 945--952, Sep. 1985, \[English translation: H. Yamamoto, “Secret sharing system using (_k_, _L_, _n_) threshold scheme,” Electronics and Communications in Japan, Part I, vol. 69, no. 9, pp. 46--54, (Scripta Technica, Inc.), Sep. 1986.\]

<a name="Blakley1985">[3]</a> G. R. Blakley, Jr. and C. Meadows, ``Security of ramp schemes,'' in Advances in Cryptology, Proceedings of CRYPTO '84, Santa Barbara, CA, USA, August 19--22, 1984, Proceedings, ser. Lecture Notes in Computer Science, G. R. Blakley, Jr. and D. Chaum, Eds., vol. 196. Heidelberg, Germany: Springer-Verlag, 1985, pp. 242--268.