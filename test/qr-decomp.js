'use strict';

var chai = require('chai');
var ndarray = require('ndarray');
var qrdecomp = require('../index.js');
var TOLERANCE = 1e-4;

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

describe('QR Decomposition', function () {
  it('4 x 3 Matrix', function () {
    var adata = [3, 2, 1,
                  2, -3, 4,
                  5, 1, -1,
                  7, 4, 2];
    var A = ndarray(new Float64Array(adata), [4, 3]);
    var qdata = [0.3216, -0.2062, -0.2511, -0.8894,
                  0.2144, 0.8989, -0.3813, -0.0232,
                  0.5361, 0.2144, 0.8121, -0.0851,
                  0.7505, -0.3216, -0.3635, 0.4486];
    var Q = ndarray(new Float64Array(qdata), [3, 3]);
    var rdata = [9.3274, 3.5380, 2.1442,
                  0, -4.1812, 2.5318,
                  0, 0, -3.3154,
                  0, 0, 0];
    var R = ndarray(new Float64Array(rdata), [3, 3]);
    var result = qrdecomp(A);
    var q = result.q;
    var r = result.r;
    chai.assert(approximatelyEqual(Q, q, TOLERANCE), 'Q matrix not within tolerance of ' + TOLERANCE + '.');
    chai.assert(approximatelyEqual(R, r, TOLERANCE), 'R matrix not within tolerance of ' + TOLERANCE + '.');
  });
});
