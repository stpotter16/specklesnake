from abc import ABC, abstractmethod


class BaseImageInterpolator(ABC):
    """ Base class for image interpolators

    Attributes
    ----------
    image : array
        A 2D ndarray representing the image as pixel values
    """
    def __init__(self, image, **kwargs):
        self.image = image

    @abstractmethod
    def __call__(row, col):
        """ Compute the interpolated value of the image at the specified
        location.

        Parameters
        ----------
        row : float
            The rowwise position of the interpolation point
        col : float
            The columwise position of the interpolation point

        Returns
        -------
        val : float
            The computed interpolation point value

        Raises
        ------
        ValueError
            row and/or col value are outside the bounds of the image
        """
        raise NotImplementedError()
