'use strict';

var ndarray = require('ndarray');
var blas1 = require('ndarray-blas-level1');

module.exports.decompose = function qrDecomp (A, q, r) {
  var givens = blas1.rotg;
  var m = A.shape[0];
  var n = A.shape[1];
  var Q = q || ndarray(new Float64Array(m * m), [m, m]);
  var R = r || ndarray(new Float64Array(m * n), [m, n]);
  var csr = new Float64Array([0, 0, 0]);

  var i = m * n;
  while (i--) {
    R.data[i] = A.data[i];
  }
  i = m;
  while (i--) {
    Q.set(i, i, 1);
  }
  var j = 0;
  for (j = 0; j < n; ++j) {
    for (i = m - 1; i >= j + 1; --i) {
      var a = R.get(i - 1, j);
      var b = R.get(i, j);
      givens(a, b, csr);
      var c = csr[0];
      var s = csr[1];
      var tmp1 = 0;
      var tmp2 = 0;

      // R' = G * R
      var x = 0;
      for (x = 0; x < n; ++x) {
        tmp1 = R.get(i - 1, x);
        tmp2 = R.get(i, x);
        R.set(i - 1, x, tmp1 * c + tmp2 * s);
        R.set(i, x, -tmp1 * s + tmp2 * c);
      }
      R.set(i - 1, j, csr[2]);
      R.set(i, j, 0);

      // Q' = Q * G^T
      for (x = 0; x < m; ++x) {
        tmp1 = Q.get(x, i - 1);
        tmp2 = Q.get(x, i);
        Q.set(x, i - 1, tmp1 * c + tmp2 * s);
        Q.set(x, i, -tmp1 * s + tmp2 * c);
      }
    }
  }

  return {
    'q': Q,
    'r': R
  };
};

module.exports.solve = function solveAxb (A, bz, x) {
  var givens = blas1.rotg;
  var m = bz.shape[0];
  var n = x.shape[0];
  if (m !== n) {
    throw new Error('No unique solution: ' + (m > n ? 'Overconstrained problem.' : 'Infinitely many solutions.'));
  }

  // Do modified QR factorization
  var R = ndarray(new Float64Array(n * n), [n, n]);
  var B = ndarray(new Float64Array(n), [n]);
  var X = x || ndarray(new Float64Array(n), [n]);
  var xo = X.offset;
  var xs = X.stride[0];
  var bo = B.offset;
  var bs = B.stride[0];
  var ro = R.offset;
  var rs0 = R.stride[0];
  var rs1 = R.stride[1];
  var csr = new Float64Array([0, 0, 0]);
  var i = n * n;
  while (i--) {
    R.data[i] = A.data[i];
  }
  i = n;
  while (i--) {
    B.data[i] = bz.data[i];
  }
  var j;
  var idx1;
  var idx2;
  for (j = 0; j < n; ++j) {
    for (i = m - 1; i >= j + 1; --i) {
      var a = R.get(i - 1, j);
      var b = R.get(i, j);
      givens(a, b, csr);
      var c = csr[0];
      var s = csr[1];
      var tmp1 = 0;
      var tmp2 = 0;

      // R' = G * R
      var k = 0;
      for (k = 0; k < n; ++k) {
        tmp1 = R.get(i - 1, k);
        tmp2 = R.get(i, k);
        R.set(i - 1, k, tmp1 * c + tmp2 * s);
        R.set(i, k, -tmp1 * s + tmp2 * c);
      }
      R.set(i - 1, j, csr[2]);
      R.set(i, j, 0);

      // b' = G * b
      idx1 = bo + (i - 1) * bs;
      idx2 = bo + i * bs;
      tmp1 = B.data[idx1];
      tmp2 = B.data[idx2];
      B.data[idx1] = tmp1 * c + tmp2 * s;
      B.data[idx2] = -tmp1 * s + tmp2 * c;
    }
  }

  for (j = n - 1; j >= 0; --j) {
    // x_j = \frac{b_j - \sum_{i=j+1}^n R_{ji} * x_i}{R_{jj}}
    idx1 = ro + (j * rs0);
    var sum = B.data[bo + j * bs];
    for (i = j + 1; i < n; ++i) {
      idx2 = xo + i * xs;
      sum -= R.data[idx1 + i * rs1] * X.data[idx2];
    }
    idx1 += j * rs1;
    idx2 = xo + j * xs;
    X.data[idx2] = sum / R.data[idx1];
  }
  return X;
};
