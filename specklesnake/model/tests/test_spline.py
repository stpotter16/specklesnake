import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from specklesnake.util.spline_knot import generate_uniform
from specklesnake.model.spline import BSplineSurface


class TestBSplineSurface(unittest.TestCase):
    def setUp(self):
        super().setUp()
        surface = BSplineSurface(degree_u=2, degree_v=2)
        surface.num_ctrlpts_u = 5
        surface.num_ctrlpts_v = 5

        x = y = np.arange(5.0)
        ys, xs = np.meshgrid(x, y)
        ctrlpt_array = np.column_stack([xs.flatten(), ys.flatten()])
        surface.control_points = ctrlpt_array

        uvec = generate_uniform(surface.degree_u, surface.num_ctrlpts_u)
        vvec = generate_uniform(surface.degree_v, surface.num_ctrlpts_v)
        surface.knot_vector_u = uvec
        surface.knot_vector_v = vvec

        self.surface = surface

    def test_surface_degree(self):
        self.assertEqual(self.surface.degree_u, 2)
        self.assertEqual(self.surface.degree_v, 2)

    def test_surface_number_of_control_points(self):
        self.assertEqual(self.surface.num_ctrlpts_u, 5)
        self.assertEqual(self.surface.num_ctrlpts_v, 5)

    def test_surface_control_points(self):
        x = y = np.arange(5.0)
        ys, xs = np.meshgrid(x, y)
        ctrlpt_array = np.column_stack([xs.flatten(), ys.flatten()])
        assert_array_almost_equal(ctrlpt_array, self.surface.control_points)

    def test_surface_knot_vectors(self):
        uvec = generate_uniform(self.surface.degree_u, self.surface.num_ctrlpts_u)
        assert_array_almost_equal(self.surface.knot_vector_u, uvec)
        vvec = generate_uniform(self.surface.degree_v, self.surface.num_ctrlpts_v)
        assert_array_almost_equal(self.surface.knot_vector_v, vvec)

    def test_surface_control_point_guard(self):
        ctrlpt_array = self.surface.control_points

        new_surface = BSplineSurface(degree_u=2)
        with self.assertRaises(ValueError):
            new_surface.control_points = ctrlpt_array

        new_surface = BSplineSurface(degree_v=2)
        with self.assertRaises(ValueError):
            new_surface.control_points = ctrlpt_array

    def test_surface_knot_vector_guard(self):
        ctrlpt_array = self.surface.control_points

        new_surface = BSplineSurface(degree_u=2, degree_v=2)
        knot_vec = generate_uniform(new_surface.degree_u, self.surface.num_ctrlpts_u)
        with self.assertRaises(ValueError):
            new_surface.knot_vector_u = knot_vec

        new_surface.control_points = ctrlpt_array
        with self.assertRaises(ValueError):
            new_surface.knot_vector_u = knot_vec

        new_surface = BSplineSurface(degree_u=2, degree_v=2)
        knot_vec = generate_uniform(new_surface.degree_v, self.surface.num_ctrlpts_v)
        with self.assertRaises(ValueError):
            new_surface.knot_vector_v = knot_vec

        new_surface.control_points = ctrlpt_array
        with self.assertRaises(ValueError):
            new_surface.knot_vector_v = knot_vec

    def test_surface_single_point(self):
        cases = [
            (0.0, 0.0, (0.0, 0.0)),
            (0.25, 0.25, (1.21875, 1.21875)),
            (0.75, 0.75,  (2.78125, 2.78125)),
            (0.5, 0.25, (2.0, 1.21875)),
            (0.25, 0.5, (1.21875, 2.0)),
            (1.0, 1.0,  (4.0, 4.0))
        ]
        for u_val, v_val, expected in cases:
            point = self.surface.single_point(u_val, v_val)
            assert_array_almost_equal(point, expected)

    def test_surface_points(self):
        knot_vals = np.array(
            [[0.0, 0.0],
            [0.25, 0.25],
            [0.75, 0.75],
            [0.5, 0.25],
            [0.25, 0.5],
            [1.0, 1.0]]
        )
        points = self.surface.points(knot_vals)
        expected = np.array(
            [[0.0, 0.0],
            [1.21875, 1.21875],
            [2.78125, 2.78125],
            [2.0, 1.21875],
            [1.21875, 2.0],
            [4.0, 4.0]]
        )
        assert_array_almost_equal(points, expected)

    def test_surface_derivatives_point(self):
        cases = [
            (0.0, 0.0, (0.0, 0.0)),
            (0.25, 0.25, (1.21875, 1.21875)),
            (0.75, 0.75,  (2.78125, 2.78125)),
            (0.5, 0.25, (2.0, 1.21875)),
            (0.25, 0.5, (1.21875, 2.0)),
            (1.0, 1.0,  (4.0, 4.0))
        ]
        for u_val, v_val, expected in cases:
            derivatives = self.surface.derivatives(u_val, v_val, 2, 2, normalize=False)
            point = derivatives[0, :]
            assert_array_almost_equal(expected, point)

    def test_surface_derivatives_u1(self):
        cases = [
            (0.0, 0.0,  (6.0, 0.0)),
            (0.25, 0.25, (3.75, 0.0)),
            (0.75, 0.75,  (3.75, 0.0)),
            (0.5, 0.25, (3.0, 0.0)),
            (0.25, 0.5, (3.75, 0.0)),
            (1.0, 1.0,  (6.0, 0.0))
        ]
        for u_val, v_val, expected in cases:
            derivatives = self.surface.derivatives(u_val, v_val, 2, 2, normalize=False)
            u1 = derivatives[3, :]
            assert_array_almost_equal(expected, u1)

    def test_surface_derivatives_u2(self):
        cases = [
            (0.0, 0.0,  (-9.0, 0.0)),
            (0.25, 0.25, (-9.0, 0.0)),
            (0.75, 0.75,  (9.0, 0.0)),
            (0.5, 0.25, (0.0, 0.0)),
            (0.25, 0.5, (-9.0, 0.0)),
            (1.0, 1.0,  (9.0, 0.0))
        ]
        for u_val, v_val, expected in cases:
            derivatives = self.surface.derivatives(u_val, v_val, 2, 2, normalize=False)
            u2 = derivatives[6, :]
            assert_array_almost_equal(expected, u2)

    def test_surface_derivatives_v1(self):
        cases = [
            (0.0, 0.0,  (0.0, 6.0)),
            (0.25, 0.25, (0.0, 3.75)),
            (0.75, 0.75,  (0.0, 3.75)),
            (0.5, 0.25, (0.0, 3.75)),
            (0.25, 0.5, (0.0, 3.0)),
            (1.0, 1.0,  (0.0, 6.0))
        ]
        for u_val, v_val, expected in cases:
            derivatives = self.surface.derivatives(u_val, v_val, 2, 2, normalize=False)
            v1 = derivatives[1, :]
            assert_array_almost_equal(expected, v1)

    def test_surface_derivatives_v2(self):
        cases = [
            (0.0, 0.0,  (0.0, -9.0)),
            (0.25, 0.25, (0.0, -9.0)),
            (0.75, 0.75,  (0.0, 9.0)),
            (0.5, 0.25, (0.0, -9.0)),
            (0.25, 0.5, (0.0, 0.0)),
            (1.0, 1.0,  (0.0, 9.0))
        ]
        for u_val, v_val, expected in cases:
            derivatives = self.surface.derivatives(u_val, v_val, 2, 2, normalize=False)
            v2 = derivatives[2, :]
            assert_array_almost_equal(expected, v2)

    def test_surface_derivatives_uv(self):
        cases = [
            (0.0, 0.0,  (0.0, 0.0)),
            (0.25, 0.25, (0.0, 0.0)),
            (0.75, 0.75,  (0.0, 0.0)),
            (0.5, 0.25, (0.0, 0.0)),
            (0.25, 0.5, (0.0, 0.0)),
            (1.0, 1.0,  (0.0, 0.0))
        ]
        for u_val, v_val, expected in cases:
            derivatives = self.surface.derivatives(u_val, v_val, 2, 2, normalize=False)
            uv = derivatives[4, :]
            assert_array_almost_equal(expected, uv)
