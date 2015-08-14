import numpy as np
from scipy.signal import fftconvolve
from astropy.wcs import WCS
from astropy.io import fits
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import RectBivariateSpline

# Script to generate sample data

# ---- CONFIGURATION START ----

dithers = [
    # ref 1
    (0.,       0.,       0.,   True),
    (3.3333,   3.6666,   0.,   True),
    (6.6666,   6.3333,   0.,   True),

    # ref 2
    (0.,       0.,       20.,   True),
    (3.3333,   3.6666,   20.,   True),
    (6.6666,   6.3333,   20.,   True),

    # new
    (0.,       0.,       40.,   False),
    (3.3333,   3.6666,   40.,   False),
    (6.6666,   6.3333,   40.,   False),
]
size_x = 100
size_y = 150
pixel_scale = 0.13

center_ra = 30.0
center_dec = 30.0

oversampling = 11.
psf_x_max = 15.
psf_y_max = 17.
prefix = 'gen'


def psf_function(x, y):
    fwhm = 1.2
    sigma = fwhm / 2.3548

    out = np.exp(-((x-0.0)**2 / (2*sigma**2/4.) + y**2 / (2*sigma**2)))
    out += np.exp(-((x)**2 / (2*sigma**2) + (y-0.5)**2 / (2*sigma**2/4.)))
    out /= np.sum(out)

    return out


def data_function(x, y):
    fwhm = 20.
    sigma = fwhm / 2.3548

    out = 0.05*np.exp(-((x+10)**2 / (2*sigma**2/2.) + y**2 / (2*sigma**2)))
    out += 0.1*np.exp(-((x-20)**2 / (2*0.1**2) + y**2 / (2*0.1**2)))
    out += 0.5*np.exp(-((x-30)**2 / (2*0.1**2) + y**2 / (2*0.1**2)))
    out += 0.2*np.exp(-((x)**2 / (2*0.1**2) + (y-5)**2 / (2*0.1**2)))

    return out

# ---- CONFIGURATION END ----#


# Generate PSF. This will be shifted so that the center of mass falls in the
# center of the image.

psf_x_range = np.arange(-psf_x_max, psf_x_max+1/2./oversampling,
                        1/oversampling)
psf_y_range = np.arange(-psf_y_max, psf_y_max+1/2./oversampling,
                        1/oversampling)

psf_x_grid, psf_y_grid = np.meshgrid(psf_x_range, psf_y_range)

# Generate data, recenter, and generate again. This prevents dumb alignment
# issues.
psf_data = psf_function(psf_x_grid, psf_y_grid)
center_y, center_x = center_of_mass(psf_data)
shift_x = center_x / oversampling - psf_x_max
shift_y = center_y / oversampling - psf_y_max
psf_data = psf_function(psf_x_grid + shift_x, psf_y_grid + shift_y)

# Write the PSF to a fits file
psf_hdu = fits.PrimaryHDU(psf_data)
psf_hdulist = fits.HDUList([psf_hdu])
psf_hdulist.writeto('./%s_psf.fits' % (prefix,), clobber=True)

# Convolve the PSF with the pixel shape. Note: we don't renormalize as we want
# to integrate over the PSF. The output is the integrated function that we
# sample.
conv_psf_data = fftconvolve(
    psf_data,
    np.ones((oversampling, oversampling)),
    mode='same'
)

# Write the convolved PSF to a fits file
conv_psf_hdu = fits.PrimaryHDU(conv_psf_data)
conv_psf_hdulist = fits.HDUList([conv_psf_hdu])
conv_psf_hdulist.writeto('./%s_conv_psf.fits' % (prefix,), clobber=True)

# Generate the data model
data_x_range = np.arange(-0.75*size_x, 0.75*size_x, 1/oversampling)
data_y_range = np.arange(-0.75*size_y, 0.75*size_y, 1/oversampling)

data_x_grid, data_y_grid = np.meshgrid(data_x_range, data_y_range)

# data = data_function(data_x_grid, data_y_grid)
# data_spline = RectBivariateSpline(data_x_range, data_y_range, data.T)

for i, dither_data in enumerate(dithers):
    x_offset, y_offset, rotation_angle, is_reference = dither_data
    print "Generating image %d" % i

    # Counter-clockwise rotation
    angle_rad = rotation_angle * np.pi / 180.
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                           [np.sin(angle_rad), np.cos(angle_rad)]])

    rot_x_grid, rot_y_grid = np.einsum(
        'ij, mni -> jmn',
        rot_matrix,
        np.dstack([data_x_grid, data_y_grid])
    )

    # Generate data and convolve with the PSF
    rot_data = data_function(rot_x_grid, rot_y_grid)
    psf_applied_data = fftconvolve(
        rot_data,
        conv_psf_data,
        mode='same'
    )

    spline = RectBivariateSpline(data_y_range, data_x_range, psf_applied_data)

    # Sample the data
    sample_x_range = np.arange(-size_x/2. + 0.5 + x_offset, size_x/2. + 0.5 +
                               x_offset)
    sample_y_range = np.arange(-size_y/2. + 0.5 + y_offset, size_y/2. + 0.5 +
                               y_offset)
    dither_data = spline(sample_y_range, sample_x_range)

    # Generate the WCS
    inv_rot_matrix = np.linalg.inv(rot_matrix)
    ra_offset, dec_offset = inv_rot_matrix.dot([x_offset, y_offset])
    dither_wcs = WCS(naxis=2)
    dither_wcs.wcs.crpix = [size_x / 2., size_y / 2.]
    dither_center_ra = (
        center_ra
        - ra_offset * pixel_scale / 3600. / np.cos(center_dec * np.pi / 180.)
    )
    dither_center_dec = center_dec + dec_offset * pixel_scale / 3600.
    dither_wcs.wcs.crval = [dither_center_ra, dither_center_dec]
    dither_wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale]) / 3600.
    dither_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    dither_wcs.wcs.pc = inv_rot_matrix

    # Write the output data to a fits file
    dither_header = dither_wcs.to_header()
    dither_header['ORIENTAT'] = rotation_angle
    dither_header['EXPTIME'] = 100.
    dither_hdu = fits.PrimaryHDU(dither_data, header=dither_header)
    dither_hdulist = fits.HDUList([dither_hdu])
    dither_hdulist.writeto('./%s_dither_%d.fits' % (prefix, i), clobber=True)