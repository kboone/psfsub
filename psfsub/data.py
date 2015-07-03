from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from scipy.interpolate import RectBivariateSpline
import astropy.units as u
from scipy.spatial import KDTree

from psfsub.psf import Psf


class Image(object):
    """Class to represent a dithered image"""
    def __init__(self, path, psf_path, psf_oversampling):
        hdulist = fits.open(path)

        self.ras = []
        self.decs = []
        self.vals = []
        self.errs = []
        self.psfs = []

        # For now, we only support WFC3 IR images. These should already have
        # the blob flats and pixel area maps applied!
        # Note: rotation angle is counterclockwise
        data = hdulist['SCI'].data
        err = hdulist['ERR'].data
        mask = hdulist['DQ'].data & ~8192
        self.rotation_angle = hdulist['SCI'].header['ORIENTAT']

        # Use a single PSF for the whole image. This can be changed to use
        # variable PSFs or something.
        psf = Psf.load(psf_path, psf_oversampling, self.rotation_angle)

        # Get a spline to do the WCS transformation. We want RA and DEC for
        # each pixel.
        self.wcs = WCS(hdulist['SCI'].header, fobj=hdulist)
        self.center_ra, self.center_dec = self.wcs.wcs.crval
        spacing = 50
        num_y, num_x = data.shape
        x_range = np.arange(0, num_x+spacing, spacing)
        y_range = np.arange(0, num_y+spacing, spacing)
        y_grid, x_grid = np.meshgrid(x_range, y_range)

        ra_grid, dec_grid = self.wcs.all_pix2world(x_grid, y_grid, 0)

        ra_spline = RectBivariateSpline(x_range, y_range, ra_grid)
        dec_spline = RectBivariateSpline(x_range, y_range, dec_grid)

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
        self.ras = ra_spline.ev(eval_x, eval_y)
        self.decs = dec_spline.ev(eval_x, eval_y)

        # KDTree. Distances need to be in arcsec
        ra_arcsec = self.ras * np.cos(self.center_dec * np.pi / 180.) * 3600.
        dec_arcsec = self.decs * 3600.
        self.kdtree = KDTree(zip(ra_arcsec, dec_arcsec))

    def find_near(self, ra, dec, match_dist):
        """Find all pixels within a given distance in arcsec"""
        # TODO: optimize this. Use an i,j box or something as an initial guess.
        # That would probably be fast, but need to figure out how to deal with
        # bad pixels and jank.

        # TODO: this could work if the ra and dec were given in 2d.
        # search_around_sky is slow for single values though.
        # compare_cat = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        # idxc, idxcat, d2d, d3d = self.catalog.search_around_sky(
        # compare_cat,
        # match_dist*u.arcsec
        # )

        #loc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        #idx = np.where(loc.separation(self.catalog) < match_dist*u.arcsec)

        ra_arcsec = ra * np.cos(self.center_dec * np.pi / 180.) * 3600.
        dec_arcsec = dec * 3600.

        idx = self.kdtree.query_ball_point((ra_arcsec, dec_arcsec), match_dist)

        return (self.vals[idx], self.errs[idx], self.psfs[idx], self.ras[idx],
                self.decs[idx])
