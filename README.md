# SSS-GF2m: Secret sharing over GF(2^m)

SS-GF2m provides a secret sharing scheme over $$GF(2^m)$$ that have been originally proposed by A. Shamir in 1979.

```
math
\begin{eqnarray}
 f(x) = \sum_{i=0}^{l-1} s_i x^i + \sum_{j=0}^{k-l-1} r_{j} x^{l+j} in \mathbb{F}_{2^m}[x]
\end{eqnarray}
```

The aim of this repo. is to provide an open source software libraries of the secret sharing scheme proposed by A. Shamir, which can be used as the benchmark of secret sharing scheme.
