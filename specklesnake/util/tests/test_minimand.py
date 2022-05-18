import unittest

import numpy as np

from specklesnake.util.minimand import zero_normalized_sum_of_square_difference


class TestZeroNormalizedSSD(unittest.TestCase):
    def setUp(self):
        # FIXME: Is it better to use a Uint8 dtype here?
        self.array_f = np.random.normal(0.0, 0.1, (100, 100))
        return super().setUp()

    def test_different_array(self):
        # Randomly mask image in frequency domain
        array_f_fourier = np.fft.fftshift(np.fft.fft2(self.array_f))
        array_f_fourier[45:50, :] = 1
        array_g = np.fft.ifft2(array_f_fourier)
        result = zero_normalized_sum_of_square_difference(
            self.array_f,
            array_g
        )
        self.assertNotAlmostEqual(0.0, result)

    def test_same_array(self):
        result = zero_normalized_sum_of_square_difference(
            self.array_f,
            self.array_f
        )
        self.assertAlmostEqual(0.0, result)

    def test_mean_shift(self):
        array_g = np.ones_like(self.array_f) + self.array_f
        result = zero_normalized_sum_of_square_difference(
            self.array_f,
            array_g
        )
        self.assertAlmostEqual(0.0, result)

    def test_gaussian_noise(self):
        array_g = self.array_f + np.random.normal(0, 0.1, self.array_f.shape)
        result = zero_normalized_sum_of_square_difference(
            self.array_f,
            array_g
        )
        self.assertAlmostEqual(0.0, result)
