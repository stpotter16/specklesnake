import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from specklesnake.model.bilinear import BilinearElement


class TestBilinearElement(unittest.TestCase):
    def setUp(self):
        super().setUp()
        element = BilinearElement()
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        element.nodes = nodes
        self.element = element

    def test_single_point(self):
        expected = [
            (-1, -1, 0),
            (1, -1, 1),
            (1, 1, 2),
            (-1, 1, 3)
        ]
        for xi, eta, node_idx in expected:
            point = self.element.single_point(xi, eta)
            assert_array_almost_equal(point, self.element.nodes[node_idx, :])

    def test_deformation_gradient(self):
        in_def_grad = np.array([
            [1.1, 0.0],
            [0.0, 1.0]
        ])
        disp_vec = self.element.nodes @ in_def_grad - self.element.nodes
        out_def_grad = self.element.deformation_gradient(0, 0, disp_vec)
        assert_array_almost_equal(in_def_grad, out_def_grad)
