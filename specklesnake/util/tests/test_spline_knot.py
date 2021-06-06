import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from specklesnake.util.spline_knot import (
    check_knot_vector, find_multiplicity, find_span, generate_uniform,
    normalize
)

SPANS = [
    (1.0 / 2.0, 2),
    (7.0 / 2.0, 5),
    (9.0 / 2.0, 7),
    (3.0, 5),
    (0.0, 2),  # Edge case: Should return left side of interval
    (3.0, 5),  # Should return left side of interval
    (5.0, 7),  # Edge case: should return num_ctrlpts - 1 (n)
]


class TestSplineKnotFunctions(unittest.TestCase):
    def test_find_span(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        num_ctrlpts = len(knot_vector) - degree - 1
        for knot_val, expected in SPANS:
            interval = find_span(num_ctrlpts, degree, knot_val, knot_vector)
            self.assertEqual(interval, expected)

    def test_normalize(self):
        original = np.array([0, 0, 0, 1, 2, 2, 2])
        normalized = np.array([0, 0, 0, 0.5, 1, 1, 1])
        assert_array_almost_equal(normalized, normalize(original))

    def test_check(self):
        degree = 2
        knot_vector = [0, 0, 0, 0.5, 1, 1, 1]
        num_ctrlpts = 4
        self.assertTrue(check_knot_vector(degree, knot_vector, num_ctrlpts))

        knot_vector = [0, 0, 0, 1, 2, 2, 2]
        self.assertTrue(check_knot_vector(degree, knot_vector, num_ctrlpts))

        num_ctrlpts = 3
        self.assertFalse(check_knot_vector(degree, knot_vector, num_ctrlpts))

    def test_generate(self):
        degree = 2
        num_ctrlpts = 4
        expected = np.array([0, 0, 0, 0.5, 1, 1, 1])
        generated = generate_uniform(degree, num_ctrlpts)
        assert_array_almost_equal(expected, generated)

        degree = 3
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        generated = generate_uniform(degree, num_ctrlpts)
        assert_array_almost_equal(expected, generated)

        degree = 2
        num_ctrlpts = 6
        expected = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
        generated = generate_uniform(degree, num_ctrlpts)
        assert_array_almost_equal(expected, generated)

    def test_multiplicity(self):
        knot_vector = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
        knot = 0.625
        self.assertEqual(find_multiplicity(knot, knot_vector), 0)

        knot = 0.5
        self.assertEqual(find_multiplicity(knot, knot_vector), 1)

        knot = 0.0
        self.assertEqual(find_multiplicity(knot, knot_vector), 3)
