'use strict';

var chai = require('chai');
var ndarray = require('ndarray');
var qr = require('../index.js');
var TOLERANCE = 1e-14;
var NUM_AUTO_TESTS = 100;

var approximatelyEqual = function (a, b, tolerance) {
  var i = a.data.length;
  while (i--) {
    var val = Math.abs(a.data[i] - b.data[i]);
    if (val > tolerance) {
      return false;
    }
  }
  return true;
};

var ndarrayToString = function (A) {
  // var a = A.data;
  var m = A.shape[0];
  var n = A.shape[1];
  var i;
  var j;
  var tabSpaces = '    ';

  var capString = '+-';
  for (j = 0; j < n - 1; ++j) {
    capString += tabSpaces + '      ';
  }
  capString += tabSpaces + '  -+\n';

  var dataString = '';
  for (i = 0; i < m; ++i) {
    var rowString = '| ';
    for (j = 0; j < n; ++j) {
      if (j !== 0) {
        rowString += tabSpaces;
      }
      var val = A.get(i, j);
      var valString;
      if (val === 0) {
        valString = '   0  ';
      } else {
        valString = val.toString().substring(0, 6);
      }
      rowString += valString;
    }
    rowString += ' |\n';
    dataString += rowString;
  }
  var retString = capString + dataString + capString;
  return retString;
};

var naiveGEMM = function (a, b, c) {
  var m = a.shape[0];
  var n = a.shape[1];
  if (b.shape[0] !== n) {
    throw new Error('GEMM: Matrices are not of proper dimensions, ' + a.shape[1] + ' != ' + b.shape[0] + '.');
  }
  var p = b.shape[1];
  var C = c || ndarray(new Float64Array(m * p), [m, p]);
  for (var i = 0; i < m; ++i) {
    for (var j = 0; j < p; ++j) {
      var sum = 0;
      var y = 0;
      var t = 0;
      var comp = 0;
      for (var k = 0; k < n; ++k) {
        var val = a.get(i, k) * b.get(k, j);
        y = val - comp;
        t = sum + y;
        comp = (t - sum) - y;
        sum = t;
      }
      C.set(i, j, sum);
    }
  }
};

describe('QR Decomposition', function () {
  it('Basic Test - 4 x 3 Matrix', function () {
    var SIMPLE_TOLERANCE = 1e-4;
    var adata = [3, 2, 1,
                  2, -3, 4,
                  5, 1, -1,
                  7, 4, 2];
    var A = ndarray(new Float64Array(adata), [4, 3]);
    var qdata = [0.3216, -0.2062, -0.2511, -0.8894,
                  0.2144, 0.8989, -0.3813, -0.0232,
                  0.5361, 0.2144, 0.8121, -0.0851,
                  0.7505, -0.3216, -0.3635, 0.4486];
    var Q = ndarray(new Float64Array(qdata), [4, 4]);
    var rdata = [9.3274, 3.5380, 2.1442,
                  0, -4.1812, 2.5318,
                  0, 0, -3.3154,
                  0, 0, 0];
    var R = ndarray(new Float64Array(rdata), [3, 3]);
    var result = qr.decompose(A);
    var q = result.q;
    var r = result.r;
    chai.assert(approximatelyEqual(Q, q, SIMPLE_TOLERANCE), 'Q matrix not within tolerance of ' + SIMPLE_TOLERANCE + '.');
    chai.assert(approximatelyEqual(R, r, SIMPLE_TOLERANCE), 'R matrix not within tolerance of ' + SIMPLE_TOLERANCE + '.');

    // check if Q*R = A
    var Aqr = ndarray(new Float64Array(12), [4, 3]);
    naiveGEMM(q, r, Aqr);
    chai.assert(approximatelyEqual(A, Aqr, SIMPLE_TOLERANCE), 'A != Q*R to tolerance of ' + SIMPLE_TOLERANCE + '.');

    // check if Q^T*Q = I; this won't be very accurate
    var identTol = 5e-4;
    var i;
    var j;
    var Qt = ndarray(new Float64Array(16), [4, 4]);
    for (i = 0; i < 4; ++i) {
      for (j = 0; j < 4; ++j) {
        Qt.set(j, i, Q.get(i, j));
      }
    }
    var I = ndarray(new Float64Array(16), [4, 4]);
    naiveGEMM(Q, Qt, I);
    for (i = 0; i < 4; ++i) {
      for (j = 0; j < 4; ++j) {
        if (i === j) {
          chai.assert(Math.abs(I.get(i, j) - 1) < identTol, 'Q not orthogonal.');
        } else {
          chai.assert(Math.abs(I.get(i, j)) < identTol, 'Q not orthogonal.');
        }
      }
    }
  });

  it(NUM_AUTO_TESTS + ' Automated Tests', function () {
    var MAX_ROWS = 8;
    var MAX_COLS = 8;
    var IDENT_TOLERANCE = 5e-4;

    var k = NUM_AUTO_TESTS;
    var A;
    var Q;
    var R;
    while (k--) {
      // test procedure
      var m = Math.floor(Math.random() * MAX_ROWS) + 1;
      var n = m + 1;
      while (n > m) {
        n = Math.floor(Math.random() * MAX_COLS) + 1;
      }
      var i;
      var j;
      A = ndarray(new Float64Array(m * n), [m, n]);
      var Aqr = ndarray(new Float64Array(m * n), [m, n]);
      var Qt = ndarray(new Float64Array(m * m), [m, m]);
      var I = ndarray(new Float64Array(m * m), [m, m]);
      var Iqqt = ndarray(new Float64Array(m * m), [m, m]);
      for (i = 0; i < m; ++i) {
        for (j = 0; j < m; ++j) {
          A.set(i, j, Math.random());
        }
      }
      var result = qr.decompose(A);
      Q = result.q;
      R = result.r;
      naiveGEMM(Q, R, Aqr);
      chai.assert(approximatelyEqual(A, Aqr, TOLERANCE), 'A != Q*R to tolerance of ' + TOLERANCE + '.');
      for (i = 0; i < m; ++i) {
        for (j = 0; j < m; ++j) {
          Qt.set(j, i, Q.get(i, j));

          if (i === j) {
            I.set(i, j, 1);
          } else {
            I.set(i, j, 0);
          }
        }
      }
      naiveGEMM(Q, Qt, Iqqt);
      var isIdentity = approximatelyEqual(I, Iqqt, IDENT_TOLERANCE);
      if (!isIdentity) {
        console.log(ndarrayToString(Iqqt));
      }
      chai.assert(isIdentity, 'Q^T*Q != I to tolerance of ' + IDENT_TOLERANCE + '.');
    }
    console.log('A = ');
    console.log(ndarrayToString(A));
    console.log('Q = ');
    console.log(ndarrayToString(Q));
    console.log('R = ');
    console.log(ndarrayToString(R));
  });
});
