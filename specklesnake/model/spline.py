import numpy as np

from specklesnake.util.spline_basis import basis_functions, basis_function_ders
from specklesnake.util.spline_knot import (
    check_knot_vector, find_span, validate_knot
)


class BSplineSurface:
    """ A B-Spline Surface

        Attributes
        ----------
        degree_u: int
            The degree of the surface in u direction
        degree_v : int
            The degree of the surface in the v direction
        control_points : array
            The control points of the surface
        knot_vector_u : array
            The knot vector of the surface in u direction
        knot_vector_v : array
            The knot voector of the surface in v direction
    """

    def __init__(self, **kwargs):
        self._degree_u = kwargs.get('degree_u', None)
        self._degree_v = kwargs.get('degree_v', None)
        self._num_control_points_u = None
        self._num_control_points_v = None
        self._control_points = None
        self._knot_vector_u = None
        self._knot_vector_v = None

    @property
    def degree_u(self):
        return self._degree_u

    @degree_u.setter
    def degree_u(self, degree):
        if degree < 1:
            raise ValueError('Degree must be greater than zero')
        self._degree_u = degree

    @property
    def degree_v(self):
        return self._degree_v

    @degree_v.setter
    def degree_v(self, degree):
        if degree < 1:
            raise ValueError('Degree must be greater than zero')
        self._degree_v = degree

    @property
    def num_ctrlpts_u(self):
        return self._num_control_points_u

    @num_ctrlpts_u.setter
    def num_ctrlpts_u(self, num):
        if num < self.degree_u + 1:
            msg = (f'Number of control points in u must be {self.degree_u + 1}'
                   f' for a surface with u degree of {self.degree_u}')
            raise ValueError(msg)

        self._num_control_points_u = num

    @property
    def num_ctrlpts_v(self):
        return self._num_control_points_v

    @num_ctrlpts_v.setter
    def num_ctrlpts_v(self, num):
        if num < self.degree_v + 1:
            msg = (f'Number of control points in v must be {self.degree_v + 1}'
                   f' for a surface with v degree of {self.degree_v}')
            raise ValueError(msg)

        self._num_control_points_v = num

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, ctrlpt_array):
        if self._degree_u is None:
            msg = 'Surface degree u must be set before setting control points'
            raise ValueError(msg)

        if self._degree_v is None:
            msg = 'Surface degree v must be set before setting control points'
            raise ValueError(msg)

        if ctrlpt_array.ndim != 2:
            raise ValueError('Control point points must be in R2')

        self._control_points = ctrlpt_array

    @property
    def knot_vector_u(self):
        return self._knot_vector_u

    @knot_vector_u.setter
    def knot_vector_u(self, kv):
        if self._degree_u is None:
            msg = ('Surface degree in u direction must be set before setting '
                   'knot vector')
            raise ValueError(msg)

        if self._control_points is None:
            msg = ('Surface control points must be set before setting knot'
                   'vector')
            raise ValueError(msg)

        if self._num_control_points_u is None:
            msg = ('Surface control point number in u must be set before '
                   'setting knot vector')
            raise ValueError(msg)

        if self._check_knot_vector(kv, direction='u'):
            self._knot_vector_u = kv

    @property
    def knot_vector_v(self):
        return self._knot_vector_v

    @knot_vector_v.setter
    def knot_vector_v(self, kv):
        if self._degree_v is None:
            msg = ('Surface degree in v direction must be set before setting '
                   'knot vector')
            raise ValueError(msg)

        if self._control_points is None:
            msg = ('Surface control points must be set before setting knot '
                   'vector')
            raise ValueError(msg)

        if self._num_control_points_v is None:
            msg = ('Surface control point number in v must be set before '
                   'setting knot vector')
            raise ValueError(msg)

        if self._check_knot_vector(kv, direction='v'):
            self._knot_vector_v = kv

    def single_point(self, u, v):
        """ Evaluate a surface at a single parametric point

        Parameters
        ----------
        knot_u : float
            Parameter in u at which to evaluate surface
        knot_v : float
            Parameter in v at withch to evaluate surface

        Returns
        -------
        point : array
            Evalued coordinate point
        """
        u = float(u)
        v = float(v)

        if not validate_knot(u):
            raise ValueError('u parameter must be in interval [0, 1]')
        if not validate_knot(v):
            raise ValueError('v parameter must be in interval [0, 1]')

        u_span = find_span(self._num_control_points_u, self._degree_u, u,
                           self._knot_vector_u)
        v_span = find_span(self._num_control_points_v, self._degree_v, v,
                           self._knot_vector_v)
        basis_funs_u = basis_functions(u_span, u, self._degree_u,
                                       self._knot_vector_u)
        basis_funs_v = basis_functions(v_span, v, self._degree_v,
                                       self._knot_vector_v)
        ctrlpt_x = self._control_points[:, 0]
        ctrlpt_y = self._control_points[:, 1]

        new_shape = (self._num_control_points_u, self._num_control_points_v)
        x_array = np.reshape(ctrlpt_x, new_shape)
        y_array = np.reshape(ctrlpt_y, new_shape)

        u_start, u_stop = u_span - self._degree_u, u_span + 1
        v_start, v_stop = v_span - self._degree_v, v_span + 1
        x_ctrlpts = x_array[u_start:u_stop, v_start:v_stop]
        y_ctrlpts = y_array[u_start:u_stop, v_start:v_stop]
        x = basis_funs_u @ x_ctrlpts @ basis_funs_v
        y = basis_funs_u @ y_ctrlpts @ basis_funs_v

        point = np.array([x, y])

        return point

    def points(self, knot_array):
        """ Evaluate the surface at multiple parametric locations

        Parameters
        ----------
        knots : array
            Array of parametric points (u,v) to evaluate

        Returns
        -------
        points : array
            Evaluated coordinat points
        """
        knot_array = np.asarray(knot_array, dtype=np.double)
        if knot_array.ndim != 2:
            raise ValueError('Parameter array must be 2D')

        values = [self.single_point(parameter[0], parameter[1]) for
                  parameter in knot_array]

        return np.array(values)

    def derivatives(self, u, v, order_u, order_v, normalize=True):
        """ Evaluate the derivatives of the surface at specified knot up to
        min (order, degree)

        Parameters
        ----------
        knot_u : float
            Parametric in u point to evaluate
        knot_v : float
            Parametric in u point to evaluate
        order_u : int
            Max order of derivatives in u to evaluate
        order_v : int
            Max order of derivatives in v to evaluate
        normalize : bool, optional
            Normalize output derivatives

        Returns
        -------
        derivs : array
            Array of points and derivatives at specified knot
        """
        u = float(u)
        v = float(v)

        if not validate_knot(u):
            raise ValueError('u parameter must be in interval [0, 1]')
        if not validate_knot(v):
            raise ValueError('v parameter must be in interval [0, 1]')

        max_order_u = min(order_u, self._degree_u)
        max_order_v = min(order_v, self._degree_v)

        u_span = find_span(self._num_control_points_u, self._degree_u, u,
                           self._knot_vector_u)
        v_span = find_span(self._num_control_points_v, self._degree_v, v,
                           self._knot_vector_v)

        basis_funs_u_ders = basis_function_ders(u_span, u, self._degree_u,
                                                self._knot_vector_u,
                                                max_order_u)
        basis_funs_v_ders = basis_function_ders(v_span, v, self._degree_v,
                                                self._knot_vector_v,
                                                max_order_v)

        ctrlpt_x = self._control_points[:, 0]
        ctrlpt_y = self._control_points[:, 1]

        new_shape = (self._num_control_points_u, self._num_control_points_v)
        x_array = np.reshape(ctrlpt_x, new_shape)
        y_array = np.reshape(ctrlpt_y, new_shape)

        u_start, u_stop = u_span - self._degree_u, u_span + 1
        v_start, v_stop = v_span - self._degree_v, v_span + 1
        x_active = x_array[u_start:u_stop, v_start:v_stop]
        y_active = y_array[u_start:u_stop, v_start:v_stop]

        derivs = np.zeros(((max_order_u + 1) * (max_order_v + 1), 2))

        index = 0
        for u_row in range(0, max_order_u + 1):
            u_ders = basis_funs_u_ders[:, u_row]
            for v_row in range(0, max_order_v + 1):

                v_ders = basis_funs_v_ders[:, v_row]
                x = u_ders @ x_active @ v_ders
                y = u_ders @ y_active @ v_ders

                val = np.array([x, y])

                condition = (normalize and
                             not np.isclose(np.linalg.norm(val), 0.0) and
                             index != 0)
                if condition:
                    val = val / np.linalg.norm(val)

                derivs[index, :] = val

                index += 1

        return derivs

    def _check_knot_vector(self, kv, direction='u'):
        """ Check that knot vector is valid
        """
        if direction == 'u':
            check = check_knot_vector(self._degree_u, kv,
                                      self._num_control_points_u)
        else:
            check = check_knot_vector(self._degree_v, kv,
                                      self._num_control_points_v)
        return check
