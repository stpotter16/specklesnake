import unittest

import numpy as np

from specklesnake.model.bicubic_image_interpolator import (
    BicubicImageInterpolator
)


class TestBicubicImageInterpolator(unittest.TestCase):
    def setUp(self):
        x = np.linspace(-6, 6, 11)
        y = np.linspace(-6, 6, 11)

        xs, ys = np.meshgrid(x, y)
        ys = np.flipud(ys)  # Y positive down

        image = self._test_function(xs, ys)
        self.image = image
        self.interpolator = BicubicImageInterpolator(image)
        return super().setUp()

    def test_pixel_values(self):
        points = [(0, 0), (5, 5), (9, 9)]
        for point in points:
            x, y = point
            expected = self.image[x, y]
            interpolated = self.interpolator(x, y)
            print(f'({x}, {y}) - {expected} vs {interpolated}')
            self.assertAlmostEqual(expected, interpolated)

    def _test_function(self, x, y):
        return np.sin(np.sqrt(x**2 + y**2))
