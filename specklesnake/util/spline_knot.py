import numpy as np


def find_span(num_ctrlpts, degree, knot, knot_vector, **kwargs):
    """
    Algorithm A2.1 from Piegl & Tiller, The NURBS Book, 1997

    Find the knot span index in knot vector for a given knot value

    Parameters
    ---------
    num_ctrlpts: int
        Number of control points in parametric direction
    degree : int
        Degree of parametric direction
    knot : float
        The value of a knot for which to find the span
    knot_vector : array
        The knot vector to search in

    Returns
    -------
    mid : int
        Index of the knot interval containing the knot
    """

    # NOTE: Number of knot intervals, n, are based on number of control points.
    # Per convention in The NURBS Book, num_ctrlpts = n + 1

    # Edge case: Return highest knot interval (n) if the knot is equal to the knot vector value
    # in that span
    rtol = kwargs.get('rtol', 1e-6)
    if np.allclose(knot, knot_vector[num_ctrlpts], rtol=rtol):
        return num_ctrlpts - 1

    # Begin binary search
    low = degree
    high = num_ctrlpts
    mid_sum = low + high
    # If even, mid value is straight forward average
    if mid_sum % 2 == 0:
        mid = mid_sum / 2
    # If odd, add 1 to mid_sum to make even, then divide
    else:
        mid = (mid_sum + 1) / 2
    mid = int(mid)

    while knot < knot_vector[mid] or knot > knot_vector[mid + 1]:
        if knot < knot_vector[mid]:
            high = mid
        else:
            low = mid
        mid = int((low + high) / 2)

    return mid

def normalize(knot_vector):
    """
    Normalize input knot vector to [0, 1].

    Parameters
    ----------
    knot_vector : array
        Knot vector to be normalized
    
    Returns
    normalized: array
        Normalized knot vector
    """
    max_knot = np.max(knot_vector)
    min_knot = np.min(knot_vector)

    normalized = knot_vector - min_knot * np.ones(knot_vector.shape)
    normalized *= 1 / (max_knot - min_knot)

    return normalized 


def check_knot_vector(degree, knot_vector, num_ctrlpts):
    """
    Confirm that the knot vector conforms the the rules for B Splines

    Parameters
    ----------
    degree : int
        Degree of spline basis
    knot_vector : array
        Knot vector in question
    num_ctrlpts : int
        Number of control points associated with knot vector
    
    Returns
    -------
    bool
        Whether or not the knot vector is valid
    """
    knot_vector = np.array(knot_vector)
    # Normalize knot vector
    knot_vector = normalize(knot_vector)
    # Check that the length is correct
    if len(knot_vector) != num_ctrlpts + degree + 1:
        return False
    # Check that the first degree + 1 values are zero
    if not np.allclose(np.zeros(degree + 1), knot_vector[:degree + 1]):
        return False
    # Check that the last degree + 1 values are 1
    if not np.allclose(np.ones(degree + 1), knot_vector[-1 * (degree + 1)]):
        return False
    # Check that the knots are increasing
    previous_knot = knot_vector[0]
    for knot in knot_vector:
        if knot < previous_knot:
            return False
        previous_knot = knot
    return True


def generate_uniform(degree, num_ctrlpts):
    """
    Generates uniform, clamped knot vector on [0, 1] given basis degree and number of control points.

    Parameters
    ----------
    degreee : int
        Degree of spline basis
    num_ctrlpts : int
        Number of control points

    Returns
    -------
    knot_vector : array
        Generated knot vector
    """
    # Specify length of total knot vector
    length_knot_vector = num_ctrlpts + degree + 1
    # Subract off the repeated knots and beginning and end
    num_middle_knots = length_knot_vector - 2 * degree
    # Create evenly spaced knots from 0 to 1 of number num_middle_knots
    middle_knot_vector = np.linspace(0, 1, num_middle_knots)
    # Append middle knot vector with repeated knots on beginning and end
    knot_vector = np.concatenate((np.zeros(degree), middle_knot_vector, np.ones(degree)))
    return knot_vector


def find_multiplicity(knot, knot_vector):
    """
    Helper function for finding the multiplicity of a given knot in a given knot vector

    Parameters
    ----------
    knot : float
        The parameter value
    knot_vector : array
        The knot vector to search in

    Returns
    -------
    mult : int
        The multiplicity of the knot
    """
    mult = 0
    for knot_span in knot_vector:
        if np.isclose(knot, knot_span):
            mult += 1
    return mult


def curve_knot_insertion(degree, old_knot_vector, old_ctrlpts, inserted_knot, num_inserts=1):
    """
    Algorithm A5.1, The NURBS Book, 1997

    Inserts knot found in knot span of old knot vector with given multiplicity a given number of times and returns the
    new knot vector

    Parameters
    ----------
    degree : int
        Degree of curve
    old_knot_vector : array
        Original curve knot vector
    old_ctrlpts : array
        Original curve control points
    inserted_knot : float
        Knot to be inserted
    num_inserts : int
        Number of times to insert the knot

    Returns
    -------
    new_knot_vector : array
        The new knot vector
    new_ctrlpts : array
        The new list of control points
    """

    # Find span and multiplicity
    inserted_knot_span = find_span(len(old_ctrlpts), degree, inserted_knot, old_knot_vector)

    knot_multiplicity = find_multiplicity(inserted_knot, old_knot_vector)

    # Knot vector lengths
    old_knot_vector_length = len(old_ctrlpts) + degree + 1
    new_knot_vector_length = len(old_knot_vector) + num_inserts

    # Create new knot vector and control point array
    new_knot_vector = np.zeros(new_knot_vector_length)
    new_ctrlpts = np.zeros((len(old_ctrlpts) + num_inserts, 3))
    R = np.zeros((degree + 1, 3))

    # Load new values
    for i in range(0, inserted_knot_span + 1):
        new_knot_vector[i] = old_knot_vector[i]

    for i in range(1, num_inserts + 1):
        new_knot_vector[inserted_knot_span + i] = inserted_knot

    for i in range(inserted_knot_span + 1, old_knot_vector_length):
        new_knot_vector[i + num_inserts] = old_knot_vector[i]

    # Save unaltered control points
    for i in range(0, inserted_knot_span - degree + 1):
        new_ctrlpts[i, :] = old_ctrlpts[i, :]

    for i in range(inserted_knot_span - knot_multiplicity, len(old_ctrlpts)):
        new_ctrlpts[i + num_inserts] = old_ctrlpts[i, :]

    for i in range(0, degree - knot_multiplicity + 1):
        R[i, :] = old_ctrlpts[inserted_knot_span - degree + i, :]

    # Insert knot
    for j in range(1, num_inserts + 1):
        L = inserted_knot_span - degree + j

        for i in range(0, degree - j - knot_multiplicity + 1):
            alpha = (inserted_knot - old_knot_vector[L + i]) / (old_knot_vector[i + inserted_knot_span + 1]
                                                                - old_knot_vector[L + i])
            R[i, :] = alpha * R[i + 1, :] + (1.0 - alpha) * R[i, :]

        new_ctrlpts[L, :] = R[0, :]
        new_ctrlpts[inserted_knot_span + num_inserts - j - knot_multiplicity, :] = R[degree - j - knot_multiplicity, :]

    # Load new the rest of the control points
    L = inserted_knot_span - degree + num_inserts
    for i in range(L + 1, inserted_knot_span - knot_multiplicity):
        new_ctrlpts[i, :] = R[i - L, :]

    return (new_knot_vector, new_ctrlpts)


def validate_knot(knot):
    """ Confirm a knot is in the range [0, 1]

    Parameters
    ----------
    knot : float
        Parameter to verify

    Returns
    -------
    bool
        Whether or not the knot is valid
    """

    return (0.0 <= knot <= 1.0)