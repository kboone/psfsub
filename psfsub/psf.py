import logging
import os
import time
import weakref
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
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
    def load(cls, path, pixel_scale_x, pixel_scale_y, oversampling,
             rotation_angle):
        """Open a PSF file.

        We use a cache so that opening the same PSF file multiple times
        actually returns the same object.

        For now, the oversampling of the PSF must be specified. This is the
        number of samples per pixel.

        rotation_angle specifies the rotation of the image counterclockwise in
        degrees.
        """
        # Round the rotation angle to 0.1 degrees. That is accurate enough, and
        # speeds things up by elimination a lot of minor rotations.
        rotation_angle = np.round(rotation_angle, 1)

        # We optimize and return copies of the same object. This is done by
        # maintaining a cache of opened objects, and checking if it is in
        # there.
        if (path, rotation_angle) in cls.__open_files:
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
        print "Loading PSF %s with rotation %.1f deg" % (path, rotation_angle)
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

        # Generate a grid of coordinates in arcsecond space
        center_y, center_x = center_of_mass(data)
        y_pix_range = ((np.arange(data.shape[0]) - center_y) * pixel_scale_y /
                       oversampling)
        x_pix_range = ((np.arange(data.shape[1]) - center_x) * pixel_scale_x /
                       oversampling)
        x_pix_grid, y_pix_grid = np.meshgrid(x_pix_range, y_pix_range)
        pixel_scale = (pixel_scale_x + pixel_scale_y) / 2.

        # Generate an interpolation spline
        spline = RectBivariateSpline(y_pix_range, x_pix_range, convolved_data)

        # Generate the rotated data on a regular grid. These can be
        # configured. Note that the units are in arcseconds, so we don't have
        # to worry about spherical trig corrections.
        # TODO: Do the bounds on this better. I want to avoid extrapolating,
        # but I'm throwing away a bunch of data right now. Just set to 0
        # outside of the range or something like that.
        x_range = np.arange(-10.*pixel_scale, 10.0001*pixel_scale,
                            pixel_scale / oversampling)
        y_range = np.arange(-10.*pixel_scale, 10.0001*pixel_scale,
                            pixel_scale / oversampling)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        angle_rad = rotation_angle * np.pi / 180.
        rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                               [np.sin(angle_rad), np.cos(angle_rad)]])

        sample_x_grid, sample_y_grid = np.einsum(
            'ij, mni -> jmn',
            rot_matrix,
            np.dstack([x_grid, y_grid])
        )

        psf.data = spline.ev(sample_y_grid, sample_x_grid)
        psf.psf_spline = RectBivariateSpline(
            y_range,
            x_range,
            psf.data
        )
        psf.pixel_scale = pixel_scale
        psf.oversampling = oversampling
        psf.x_range = x_range
        psf.y_range = y_range
        psf.x_grid = x_grid
        psf.y_grid = y_grid

        psf.path = path
        psf.rotation_angle = rotation_angle

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

        #spline = RegularGridInterpolator((ra_range, dec_range),
                                         #convolved_data.T)
        spline = RectBivariateSpline(y_range, x_range, convolved_data)

        return spline

    def get_taylor_spline(self, order_x, order_y):
        """Get the convolution of the PSF with (x^order_x*y^order_y)"""

        data1 = self.data
        data2 = self.x_grid**order_x * self.y_grid**order_y

        convolved_data = fftconvolve(
            data1,
            data2,
            mode='same'
        ) / self.oversampling / self.oversampling

        spline = RectBivariateSpline(self.y_range, self.x_range,
                                     convolved_data)

        return spline

    def __str__(self):
        return "%s - %.1f deg" % (os.path.basename(self.path),
                                  self.rotation_angle)
