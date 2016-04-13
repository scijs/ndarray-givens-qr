'use strict';

var ndarray = require('ndarray');
var blas1 = require('ndarray-blas-level1');

module.exports = function qrDecomp (A) {
  var givens = blas1.rotg;
  var m = A.shape[0];
  var n = A.shape[1];
  var Q = ndarray(new Float64Array(m * m), [m, m]);
  var R = ndarray(new Float64Array(m * n), [m, n]);
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

      // R = G^T * R
      var x = 0;
      for (x = 0; x < n; ++x) {
        tmp1 = R.get(i - 1, x);
        tmp2 = R.get(i, x);
        R.set(i - 1, x, tmp1 * c + tmp2 * s);
        R.set(i, x, -tmp1 * s + tmp2 * c);
      }
      // Q = Q * G
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
