# ndarray-givens-qr
QR decomposition using Givens rotations.

## Usage

### Solving Ax=b Problems

```
var qrDecomp = require('ndarray-givens-qr');
var ndarray = require('ndarray');

var A = ndarray(new Float64Array([...]), [m, n]);
var b = ndarray(new Float64Array([...]), [m]);
var x = ndarray(new Float64Array([...]), [n]);

qr.solveAxb(A, b, x);
```

Alternatively, if no output vector is specified, then the vector is created and returned as shown:

```
var qrDecomp = require('ndarray-givens-qr');
var ndarray = require('ndarray');

var A = ndarray(new Float64Array([...]), [m, n]);
var b = ndarray(new Float64Array([...]), [m]);

var x = qr.solveAxb(A, b);
```

### Matrix Factorization

```
var qrDecomp = require('ndarray-givens-qr');
var ndarray = require('ndarray');

var A = ndarray(new Float64Array([...]), [m, n]);
var Q = ndarray(new Float64Array([...]), [m, m]);
var R = ndarray(new Float64Array([...]), [m, n]);

qr.decompose(A, Q, R);
```

Alternatively, if no output matrices are specified, then proper matrices are created and returned as shown:

```
var qr = require('ndarray-givens-qr');
var ndarray = require('ndarray');

var A = ndarray(new Float64Array([...]), [m, n]);

var results = qr.decompose(A);
var Q = results.q;
var R = results.r;
```

## Background

For a matrix A with m rows and n columns, QR decompositions create an `m x m` matrix Q and an `m x n` matrix R, where Q is a unitary matrix and R is upper triangular.

The idea behind using Givens rotations is clearing out the zeros beneath the diagonal entries of A. A rotation matrix that rotates a vector on the X-axis can be rearranged to the following form:

```
+-     -+ +- -+   +- -+
| c  -s | | a | = | r |
| s   c | | b |   | 0 |
+-     -+ +- -+   +- -+
```

This rearranging effectively "clears out" the last entry in the column vector. Givens rotations generalize this so that any row i and j can be "cleared out".

## Algorithm

The algorithm computes the Givens rotation using [BLAS Level 1](https://www.github.com/scijs/ndarray-blas-level1). The Givens rotation matrix is:

```
             +-                         -+
G(i,j,c,s) = | 1  ...  0  ...  0  ...  0 |
             |                           |
             | |   \   |       |       | |
             |                           |
             | 0  ...  c  ...  s  ...  0 |
             |                           |
             | |       |   \   |       | |
             |                           |
             | 0  ... -s  ...  c  ...  0 |
             |                           |
             | |       |       |   \   | |
             |                           |
             | 0  ...  0  ...  0  ...  1 |
             +-                         -+
```

where `c` and `s` are the values computed from the ROTG from BLAS Level 1. These values only appear at the intersection of the ith and jth rows and columns.

In pseudocode, the algorithm is:

```
R = A
Q = I
for j = 0 : 1 : n-1
  for i = m-1 : -1 : j+1
    a = A(i-1, j)
    b = A(i, j)
    [c, s, r] = BLAS1.rotg(a, b)
    G = givens(i, j, c, s)
    R = G^T * R
    Q = Q * G
  end
end
```

## Future Work

This algorithm can be parallelized. Since each Givens rotation only affects the ith and jth rows of the R matrix, more than one column can be updated at a time. The stages at which a subdiagonal entry can be annihilated ("cleared out") for an 8x8 matrix is given as:

```
+-                             -+
| *                             |
| 7   *                         |
| 6   8   *                     |
| 5   7   9   *                 |
| 4   6   8   10  *             |
| 3   5   7   9   11  *         |
| 2   4   6   8   10  12  *     |
| 1   3   5   7   9   11  13  * |
+-                             -+
```