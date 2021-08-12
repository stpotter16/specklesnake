import numpy as np


class BilinearElement:
    """ A Bilinear Finite Element

    Attributes
    ----------
    nodes : array
        The nodes of the element in global coordinates
    """
    def __init__(self, **kwargs):
        self._nodes = None

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, array):
        # FIXME: Do some validation here
        self._nodes = array

    def deformation_gradient(self, xi, eta, disp_vec):
        """ Evaluate the deformation gradient at a single parametric point

        Parameters
        ----------
        xi : float
            Parameter space value in xi at which to evaluate surface
        eta : float
            Parameter space value in eta at which to evaluate surface
        disp_vect : array
            The displacement of the element's nodes

        Returns
        -------
        deformation_gradient : array
            Evalued deformation gradient
        """
        jac_inv = self._inverse_jacobian(xi, eta)
        partial_basis_xi, partial_basis_eta = self._basis_derivs(xi, eta)
        local_basis_partials = np.vstack([partial_basis_xi, partial_basis_eta])
        global_basis_partials = jac_inv @ local_basis_partials
        partial_disp_x1 = global_basis_partials @ disp_vec[:, 0]
        partial_disp_x2 = global_basis_partials @ disp_vec[:, 1]
        deformation_gradient = np.vstack([partial_disp_x1, partial_disp_x2])
        deformation_gradient += np.eye(2)
        return deformation_gradient

    def single_point(self, xi, eta):
        """ Evaluate a surface at a single parametric point

        Parameters
        ----------
        xi : float
            Parameter space value in xi at which to evaluate surface
        eta : float
            Parameter space value in eta at which to evaluate surface

        Returns
        -------
        point : array
            Evalued coordinate point
        """
        basis_vals = self._basis_vals(xi, eta)
        return basis_vals @ self.nodes

    def _basis_vals(self, xi, eta):
        return np.array([
            0.25 * (xi - 1) * (eta - 1),
            0.25 * (xi + 1) * (1 - eta),
            0.25 * (xi + 1) * (eta + 1),
            0.25 * (1 - xi) * (eta + 1)
        ])

    def _basis_derivs(self, xi, eta):
        d_xi = np.array([
            0.25 * (eta - 1),
            0.25 * (1 - eta),
            0.25 * (eta + 1),
            0.25 * (-eta - 1)
        ])
        d_eta = np.array([
            0.25 * (xi - 1),
            0.25 * (-xi - 1),
            0.25 * (xi + 1),
            0.25 * (1 - xi)
        ])
        return d_xi, d_eta

    def _inverse_jacobian(self, xi, eta):
        partial_basis_xi, partial_basis_eta = self._basis_derivs(xi, eta)
        partial_xi = partial_basis_xi @ self.nodes
        partial_eta = partial_basis_eta @ self.nodes
        jacobian = np.vstack([partial_xi, partial_eta])
        return np.linalg.inv(jacobian)
