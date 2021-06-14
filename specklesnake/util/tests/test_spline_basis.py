import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from specklesnake.util.spline_basis import (
    basis_functions, basis_function_ders, one_basis_function,
    one_basis_function_ders
)


class TestSplineBasisFunctions(unittest.TestCase):
    def test_basis_function(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        knot_span = 4
        knot = 5.0 / 2.0

        # The NURBS Book Ex. 2.3
        basis_vals = basis_functions(knot_span, knot, degree, knot_vector)
        expected = np.array([0.125, 0.75, 0.125])
        assert_array_almost_equal(basis_vals, expected)

    def test_basis_function_sum(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        knot_span = 4
        knot = 5.0 / 2.0

        # The NURBS Book Ex. 2.3
        basis_vals = basis_functions(knot_span, knot, degree, knot_vector)
        self.assertEqual(1.0, np.sum(basis_vals))

    def test_basis_function_derivatives(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        # n = m - p - 1 -> n + 1 = m + 1 - p - 1
        knot_span = 4
        knot = 5.0/2.0
        deriv_order = 2

        # The NURBS Book Ex. 2.4
        ders_vals = basis_function_ders(knot_span, knot, degree, knot_vector,
                                        deriv_order)

        expected = np.array([[0.125, -0.5, 1.0],
                            [0.75, 0, -2.0],
                            [0.125, 0.5, 1.0]])

        assert_array_almost_equal(ders_vals, expected)

    def test_one_basis_function(self):

        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        # n = m - p - 1 -> n + 1 = m + 1 - p - 1
        knot = 5.0/2.0

        # The NURBS Book Ex. 2.5
        basis_val1 = one_basis_function(degree, knot_vector, 3, knot)
        basis_val2 = one_basis_function(degree, knot_vector, 4, knot)

        basis_vals = np.array([basis_val1, basis_val2])

        expected = np.array([0.75, 0.125])

        assert_array_almost_equal(basis_vals, expected)

    def test_one_basis_function_max_span(self):
        degree = 3
        knot_vector = [0., 0., 0., 0., 1., 1., 1., 1.]
        span = degree
        knot = 0.5

        # Smoke test
        one_basis_function(degree, knot_vector, span, knot)

    def test_one_basis_function_ders(self):

        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        # n = m - p - 1 -> n + 1 = m + 1 - p - 1
        knot_span = 4
        knot = 5.0/2.0
        deriv_order = 2

        # The NURBS Book Ex. 2.4
        basis_deriv_vals = one_basis_function_ders(degree, knot_vector,
                                                   knot_span, knot,
                                                   deriv_order)

        expected = np.array([0.125, 0.5, 1.0])

        assert_array_almost_equal(basis_deriv_vals, expected)
