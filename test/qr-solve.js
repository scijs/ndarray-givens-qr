var chai = require('chai');
var ndarray = require('ndarray');
var qr = require('../index.js');
var TOLERANCE = 1e-13;
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

var gemm = function (a, b, c) {
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

describe('QR Solve', function () {
  it('Simple 3x3 Test', function () {
    var n = 3;
    var adata = [3, -0.1, -0.2,
                  0.1, 7, -0.3,
                  0.3, -0.2, 10];
    var bdata = [7.85, -19.3, 71.4];
    var xdata = [0, 0, 0];
    var b0data = [0, 0, 0];
    var A = ndarray(new Float64Array(adata), [n, n]);
    var x = ndarray(new Float64Array(xdata), [n, 1]);
    var b = ndarray(new Float64Array(bdata), [n, 1]);
    var b0 = ndarray(new Float64Array(b0data), [n, 1]);
    chai.assert(qr.solve(A, b, x), 'QR solve test should never fail.'); // x should equal [3, -2.5, 7]
    gemm(A, x, b0);
    chai.assert(approximatelyEqual(b, b0, TOLERANCE), 'b != b0 to tolerance of ' + TOLERANCE);
  });

  it(NUM_AUTO_TESTS + ' Automated Tests - No Singular Matrix Check', function () {
    var k = NUM_AUTO_TESTS;
    var MAX_COLS = 9;
    var n;
    var A;
    var x;
    var b;
    var b0;
    var i;
    while (k--) {
      n = Math.floor(Math.random() * MAX_COLS) + 1;
      A = ndarray(new Float64Array(n * n), [n, n]);
      x = ndarray(new Float64Array(n), [n, 1]);
      b = ndarray(new Float64Array(n), [n, 1]);
      b0 = ndarray(new Float64Array(n), [n, 1]);
      i = n * n;
      while (i--) {
        A.data[i] = Math.random();
      }
      i = n;
      while (i--) {
        b.data[i] = Math.random();
      }
      chai.assert(qr.solve(A, b, x), 'QR solve test should never fail.');
      gemm(A, x, b0);
      chai.assert(approximatelyEqual(b, b0, TOLERANCE), 'b != b0 to tolerance of ' + TOLERANCE);
    }
    console.log('A = ');
    console.log(ndarrayToString(A));
    console.log('b = ');
    console.log(ndarrayToString(x));
    console.log('x = ');
    console.log(ndarrayToString(b));
  });
});
