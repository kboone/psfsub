import numpy as np
from scipy.interpolate import RectBivariateSpline

from seechange.data import FitsFile

from psfsub.psf import Psf


class Image(object):
    """Class to represent a dithered image"""
    def __init__(self, path, psf_path, psf_oversampling):
        print "Loading %s" % path
        fits_file = FitsFile.open(path, readonly=True)

        self.ras = []
        self.decs = []
        self.vals = []
        self.errs = []
        self.psfs = []

        # For now, we only support WFC3 IR images. These should already have
        # the blob flats and pixel area maps applied!
        # Note: rotation angle is counterclockwise
        data = fits_file.get_data()
        err = fits_file.get_sky_error()
        mask = fits_file.get_mask()
        self.rotation_angle = fits_file.get_angle()

        # Use a single PSF for the whole image. This can be changed to use
        # variable PSFs or something.
        psf = Psf.load(psf_path, fits_file.pixel_scale, psf_oversampling,
                       self.rotation_angle)

        # Get a spline to do the WCS transformation. We want RA and DEC for
        # each pixel.
        self.wcs = fits_file.get_wcs()
        self.center_ra, self.center_dec = self.wcs.wcs.crval
        num_y, num_x = data.shape
        spacing_x = np.min([50, num_x / 10])
        spacing_y = np.min([50, num_y / 10])
        x_range = np.arange(0, num_x+spacing_x, spacing_x)
        y_range = np.arange(0, num_y+spacing_y, spacing_y)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        ra_grid, dec_grid = self.wcs.all_pix2world(x_grid, y_grid, 0)

        ra_spline = RectBivariateSpline(y_range, x_range, ra_grid)
        dec_spline = RectBivariateSpline(y_range, x_range, dec_grid)

        eval_x = []
        eval_y = []

        for i in range(num_x):
            for j in range(num_y):
                if not mask[j, i]:
                    self.vals.append(data[j, i])
                    self.errs.append(err[j, i])
                    self.psfs.append(psf)
                    eval_x.append(i)
                    eval_y.append(j)

        self.vals = np.array(self.vals)
        self.errs = np.array(self.errs)
        self.psfs = np.array(self.psfs)
        self.ras = ra_spline.ev(eval_y, eval_x)
        self.decs = dec_spline.ev(eval_y, eval_x)
