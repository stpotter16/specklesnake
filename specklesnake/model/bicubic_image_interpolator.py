import numpy as np
from scipy.ndimage import sobel

from specklesnake.model.base_image_interpolator import BaseImageInterpolator


class BicubicImageInterpolator(BaseImageInterpolator):
    C = np.array([[1., 0., 0., 0.],
                  [0., 0., 1., 0.],
                  [-3., 3., -2., -1.],
                  [2., -2., 1., 1.]])

    D = np.array([[1., 0., -3., 2.],
                  [0., 0., 3., -2.],
                  [0., 1., -2., 1.],
                  [0., 0., -1., 1.]])

    def __init__(self, image):
        self.image = image
        self.dx = None
        self.dy = None
        self.dxy = None
        self._coefficient_cache = {}

    def __call__(self, row, col):
        self._validate_point(row, col)
        xmin = int(np.floor(col))
        x = col - xmin
        ymin = int(np.floor(row))
        y = row - ymin

        if None in (self.dx, self.dy, self.dxy):
            self._compute_sobel()

        key = f'row{ymin}col{xmin}'
        coeffs = self._coefficient_cache.get(key, None)

        if coeffs is None:
            coeffs = self._compute_bicubic_coefficients(xmin, ymin)
            self._coefficient_cache[key] = coeffs

        xs = np.array([1, x, x**2, x**3])
        ys = np.array([1, y, y**2, y**3])

        return xs.dot(coeffs.dot(ys))

    def _compute_bicubic_coefficients(self, col, row):
        """
        (0, 1) x ------ x (1, 1)
               |        |
               |        |
               |        |
        (0, 0) x ------ x (1, 0)
        """
        col_slice = slice(col, col + 2, 1)
        row_slice = slice(row, row + 2, 1)

        F = np.zeros((4, 4))
        F[:2, :2] = np.flipud(self.image[col_slice, row_slice])
        F[2:, :2] = np.flipud(self.dy[col_slice, row_slice])
        F[:2, 2:] = np.flipud(self.dx[col_slice, row_slice])
        F[2:, 2:] = np.flipud(self.dxy[col_slice, row_slice])

        coeffs = self.C @ F @ self.D

        return coeffs

    def _compute_sobel(self):
        self.dx = sobel(self.image, axis=0, mode='constant')
        self.dy = sobel(self.image, axis=1, mode='constant')
        self.dxy = sobel(self.dy, axis=0, mode='constant')

    def _validate_point(self, row, col):
        pass
