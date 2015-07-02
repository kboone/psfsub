import logging
import time
import weakref
from scipy.interpolate import RectBivariateSpline
from scipy.signal import fftconvolve
from scipy.ndimage.measurements import center_of_mass
import numpy as np

from astropy.io import fits

logger = logging.getLogger(__name__)


class Psf(object):
    """Class to represent a PSF"""

    __open_files = {}

    def __init__(self):
        self._init = False

    @classmethod
    def load(cls, path, oversampling, rotation_angle):
        """Open a PSF file.

        We use a cache so that opening the same PSF file multiple times
        actually returns the same object.

        For now, the oversampling of the PSF must be specified. This is the
        number of samples per pixel.

        rotation_angle specifies the rotation of the image counterclockwise in
        degrees.
        """
        # We optimize and return copies of the same object. This is done by
        # maintaining a cache of opened objects, and checking if it is in
        # there.
        if path in cls.__open_files:
            psf = cls.__open_files[(path, rotation_angle)]()

            # This only returns a weakref, which can be None!
            if psf is not None:
                # Make sure that it is initialized.
                scale_factor = 1.0
                fail_count = 0
                while not psf._init:
                    fail_count += 1
                    if fail_count > 5:
                        logger.warn("Second instance of %s waiting for "
                                    "initialization." % path)
                    time.sleep(0.1*scale_factor)
                    scale_factor *= 1.2
                return psf

        # Create the new object
        psf = Psf()

        # Add it to the cache
        cls.__open_files[(path, rotation_angle)] = weakref.ref(psf)

        # Load the PSF and normalize it. For now, I assume that the PSF is in
        # David's format. This means that the sum over all data has been
        # normalized to 1, and that to get the pixel value at any point we have
        # to sum over a box around the desired point.
        hdulist = fits.open(path)
        data = hdulist[0].data
        convolved_data = fftconvolve(
            data,
            np.ones((oversampling, oversampling)),
            mode='same'
        )

        # Generate a grid of coordinates
        center_y, center_x = center_of_mass(data)
        y_range = (np.arange(data.shape[0]) - center_y) / oversampling
        x_range = (np.arange(data.shape[1]) - center_x) / oversampling
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Generate an interpolation spline
        spline = RectBivariateSpline(x_range, y_range, convolved_data.T)

        # Generate the rotated data on a regular grid. These can be
        # configured...
        rot_x_range = np.arange(-10., 10.0001, 1/11.)
        rot_y_range = np.arange(-10., 10.0001, 1/11.)
        rot_x_grid, rot_y_grid = np.meshgrid(rot_x_range, rot_y_range)

        angle_rad = rotation_angle * np.pi / 180.
        rot_matrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                               [-np.sin(angle_rad), np.cos(angle_rad)]])

        sample_x_grid, sample_y_grid = np.einsum(
            'ji, mni -> jmn',
            rot_matrix,
            np.dstack([rot_x_grid, rot_y_grid])
        )

        psf.psf_spline = spline
        psf.data = spline.ev(sample_x_grid, sample_y_grid)
        psf.x_range = rot_x_range
        psf.y_range = rot_y_range
        psf.x_grid = rot_x_grid
        psf.y_grid = rot_y_grid

        psf._init = True

        return psf

    def get_convolution_spline(self, other_psf):
        """Generate a convolution spline (GG, GH, HH)"""
        x_range = self.x_range
        y_range = self.y_range

        data1 = self.data
        data2 = other_psf.data[::-1, ::-1]

        convolved_data = fftconvolve(
            data1,
            data2,
            mode='same'
        )

        spline = RectBivariateSpline(x_range, y_range, convolved_data.T)

        return spline
