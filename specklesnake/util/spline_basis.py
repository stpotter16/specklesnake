import numpy as np


def basis_functions(knot_span, knot, degree, knot_vector):
    """
    Algorithm A2.2 from Piegl & Tiller, The NURBS Book, 1997

    Compute non-vanishing basis functions for a given knot value and knot index

    Parameters
    ----------
    knot_span : int
        Knot vector span containing knot
    knot : float
        Value of knot
    degree : int
        Degree of basis
    knot_vector : array
        Knot vector containing knot and knot span

    Returns
    -------
    N : array
        Non-vanishing basis functions evaluated at knot
    """

    # Initialize empty array to hold the degree + 1 non-vanishing basis values.
    # Note N[0] = 1.0 by definition
    N = np.ones(degree + 1)

    # Initialize empty array to hold left and right computation values
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)

    # Account for the fact that range goes up to max - 1
    for j in range(1, degree + 1):
        # Setup left and right values
        left[j] = knot - knot_vector[knot_span + 1 - j]
        right[j] = knot_vector[knot_span + j] - knot
        saved = 0.0

        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved

    return N


def basis_function_ders(knot_span, knot, degree, knot_vector, deriv_order):
    """
    Algorithm A2.3 from Piegl & Tiller, The NURBS Book, 1997

    Compute non-vanishing basis functions and associated derivatives up to a
    specified order for a given knot value and knot index

    Parameters
    ----------
    knot_span : int
        Knot vector span containing knot
    knot : float
        Value of knot
    degree : int
        Degree of basis
    knot_vector : array
        Knot vector containing knot and knot span
    deriv_order : int
        Highest order of derivatives to be computed. `deriv_order <= degree`

    Returns
    -------
    ders : array
        Array containing values of non-vanishing basis functions and all
        derivative orders up to `deriv_order` evalued at `knot`

    Notes
    -----
    ders array structure:
    row 0 - Values at knot_span [0th order, 1st order, ..., deriv_order]
    row 1 - Values at knot_span - 1 [0th order, 1st order, ..., deriv_order]
    .
    .
    .
    row degree - Values at knot_span - degree
    """

    # Initialize output and local arrays
    ders = np.zeros((degree + 1, deriv_order + 1))
    # Note, this deviates from the structure found in the NURBS book
    ndu = np.ones((degree + 1, degree + 1))
    ndu[0, 0] = 1.0
    left = np.ones(degree + 1)
    right = np.ones(degree + 1)
    a = np.ones((2, degree + 1))

    # Create basis function triangles
    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[knot_span + 1 - j]
        right[j] = knot_vector[knot_span + j] - knot
        saved = 0.0

        for r in range(0, j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]

            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        ndu[j, j] = saved

    # Fill in basis function values (no derivative)
    for j in range(0, degree + 1):
        ders[j, 0] = ndu[j, degree]

    # Compute derivatives
    for r in range(0, degree + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0

        # Loop to kth derivative
        for k in range(1, deriv_order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = degree - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += (a[s2, j] * ndu[rk + j, pk])
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += (a[s2, k] * ndu[r, pk])

            ders[r, k] = d

            # Swap rows of a
            j = s1
            s1 = s2
            s2 = j

    # Multiply correction factors
    r = degree
    for k in range(1, deriv_order + 1):
        for j in range(0, degree + 1):
            ders[j, k] *= r
        r *= (degree - k)

    return ders


def one_basis_function(degree, knot_vector, knot_span, knot):
    """
    Algorithm A2.4 from Piegl & Tiller, The NURBS Book, 1997

    Compute value of single basis function at specified knot span and
    knot value with given knot vector and degree.

    Parameters
    ----------
    degree : int
        Degree of basis
    knot_vector : array
        Knot vector containing knot and knot span
    knot_span : int
        Knot vector span containing knot
    knot : float
        Value of knot

    Returns
    -------
    float
        Value of specified basis function and specified knot value
    """

    # Check some special cases first.
    # Account for the fact that arrays are zero indexed
    condition = ((knot_span == 0 and knot == knot_vector[0]) or
                 (knot_span == len(knot_vector) - degree - 2 and
                 knot == knot_vector[len(knot_vector) - 1]))
    if condition:
        return 1.0

    # If knot value is outside the compact support of the basis function
    # return zero
    condition = (knot < knot_vector[knot_span] or
                 knot > knot_vector[knot_span + degree + 1])
    if condition:
        return 0.0

    # Initialize zero degree functions. Length corresponds to number of knot
    # spans in range of support
    N = np.zeros(knot_span + degree + 1)

    for j in range(0, degree + 1):
        if knot_vector[knot_span + j] <= knot < knot_vector[knot_span + j + 1]:
            N[j] = 1.0

    # Compute the table of basis functions
    for k in range(1, degree + 1):
        saved = 0.0
        if N[0] != 0.0:
            numerator = (knot - knot_vector[knot_span]) * N[0]
            demoninator = knot_vector[knot_span + k] - knot_vector[knot_span]
            saved = numerator / demoninator

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[knot_span + j + 1]
            Uright = knot_vector[knot_span + j + k + 1]

            if N[j + 1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j + 1] / (Uright - Uleft)
                N[j] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp

    return N[0]


def one_basis_function_ders(degree, knot_vector, knot_span, knot, deriv_order):
    """
    Algorithm A2.5 from Piegl & Tiller, The NURBS Book, 1997

    Compute non-vanishing basis functions and associated derivatives up to a
    specified order of a specified basis function at a give knot value

    Parameters
    ----------
    degree : int
        Degree of basis
    knot_vector : array
        Knot vector containing knot and knot span
    knot_span : int
        Knot vector span containing knot
    knot : float
        Value of knot
    deriv_order : int
        Highest order of derivatives to be computed. `deriv_order <= degree`

    Returns
    -------
    ders : array
        Array containing values of specified basis functions and all derivative
        orders up to `deriv_order` evaluated at knot value
    """

    # Initialize return variable
    ders = np.zeros(deriv_order + 1)

    # Check that knot is in support of basis function
    condition = (knot < knot_vector[knot_span] or
                 knot >= knot_vector[knot_span + degree + 1])

    if condition:
        return ders

    # Initialize variable to hold table of lower order basis values
    N = np.zeros((degree + 1, degree + 1))

    # Fill zero order values
    for j in range(0, degree + 1):
        if knot_vector[knot_span + j] <= knot < knot_vector[knot_span + j + 1]:
            N[j, 0] = 1.0

    # Fill the rest of the table
    for k in range(1, degree + 1):
        saved = 0.0
        if N[0, k - 1] != 0.0:
            numerator = (knot - knot_vector[knot_span]) * N[0, k - 1]
            denominator = knot_vector[knot_span + k] - knot_vector[knot_span]
            saved = numerator / denominator

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[knot_span + j + 1]
            Uright = knot_vector[knot_span + j + k + 1]

            if N[j + 1, k - 1] == 0.0:
                N[j, k] = saved
                saved = 0.0
            else:
                temp = N[j + 1, k - 1] / (Uright - Uleft)
                N[j, k] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp

    ders[0] = N[0, degree]  # Basis function value

    # Now compute the derivatives
    ND = np.zeros(deriv_order + 1)
    for k in range(1, deriv_order + 1):

        # Grab the right column from the table of basis function values
        # we built above
        for j in range(0, k + 1):
            ND[j] = N[j, degree - k]

        # Compute the table of lower order functions needed to build derivative
        for jj in range(1, k + 1):
            saved = 0.0
            if ND[0] != 0.0:
                numerator = ND[0]
                first_knot = knot_vector[knot_span + degree - k + jj]
                last_knot = knot_vector[knot_span]
                denominator = first_knot - last_knot
                saved = numerator / denominator

            for j in range(0, k - jj + 1):
                Uleft = knot_vector[knot_span + j + 1]
                Uright = knot_vector[knot_span + j + degree - k + jj + 1]
                if ND[j + 1] == 0.0:
                    ND[j] = (degree - k + jj) * saved
                    saved = 0.0
                else:
                    temp = ND[j + 1] / (Uright - Uleft)
                    ND[j] = (degree - k + jj) * (saved - temp)
                    saved = temp

        ders[k] = ND[0]

    return ders
