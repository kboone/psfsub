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
    (0.,       0.,       0.0,   True),
    (3.3333,   3.6666,   0.0,   True),
    (6.6666,   6.3333,   0.0,   True),

    # ref 2
    (0.,       0.,       30.0,   True),
    (3.3333,   3.6666,   30.0,   True),
    (6.6666,   6.3333,   30.0,   True),

    # new
    (0.,       0.,       60.0,   False),
    (3.3333,   3.6666,   60.0,   False),
    (6.6666,   6.3333,   60.0,   False),
]
size_x = 100
size_y = 150
pixel_scale = 0.13

center_ra = 30.0
center_dec = 30.0

noise_sigma = 0.1

oversampling = 11.
psf_x_max = 15.
psf_y_max = 17.
prefix = 'gen'


psf_mode = 1


def gauss2d(amp, center_x, center_y, sigma_x, sigma_y, x, y):
    norm = amp / (2. * np.pi * sigma_x * sigma_y) / oversampling / oversampling
    func = np.exp(-((x-center_x)**2 / (2.*sigma_x**2) + (y-center_y)**2 /
                    (2.*sigma_y**2)))
    return norm * func

if psf_mode == 0:
    # Made up asymmetrical PSF
    def psf_function(x, y):
        fwhm = 1.2
        sigma = fwhm / 2.3548

        #out = np.exp(-x**2 / (2*sigma**2) + -y**2 / (2*sigma**2))
        out = np.exp(-((x-0.5)**2 / (2*sigma**2/4.) + y**2 / (2*sigma**2)))
        #out += np.exp(-((x)**2 / (2*sigma**2) + (y-0.5)**2 / (2*sigma**2/4.)))
        out /= np.sum(out)

        return out

elif psf_mode == 1:
    # Load the PSF from a file instead.
    hdulist = fits.open(
        '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits'
    )
    psf_read_data = hdulist[0].data

    psf_len_x, psf_len_y = psf_read_data.shape

    psf_read_x_range = ((np.arange(psf_len_x) - psf_len_x / 2.) / 11.)
    psf_read_y_range = ((np.arange(psf_len_x) - psf_len_x / 2.) / 11.)

    psf_read_spline = RectBivariateSpline(psf_read_y_range, psf_read_x_range,
                                          psf_read_data)

    def psf_function(x, y):
        return psf_read_spline.ev(y, x)

stars = []


def add_star(data, x, y, star_x, star_y, star_flux):
    # Don't use fast mode. It's not that much faster and it is much more
    # inaccurate!
    fast = False
    if fast:
        sigma = 1. / oversampling / 5.
        star_data = gauss2d(star_flux, star_x, star_y, sigma, sigma, x, y)
        star_data = star_data * star_flux / np.sum(star_data)
        data += star_data
    else:
        stars.append((star_x, star_y, star_flux))


def data_function(x, y, is_reference):
    global stars
    stars = []

    out = np.zeros(x.shape)
    #fwhm = 20.
    #sigma = fwhm / 2.3548
    #out = 0.05*np.exp(-((x+10)**2 / (2*sigma**2/2.) + y**2 / (2*sigma**2)))

    #fwhm = 4.
    #sigma = fwhm / 2.3548
    #out += 0.05*np.exp(-((x+20)**2 / (2*sigma**2/2.) + y**2 / (2*sigma**2)))

    #fwhm = 2.
    #sigma = fwhm / 2.3548
    #out += 0.1*np.exp(-((x+20)**2 / (2*sigma**2/2.) + (y-20)**2 / (2*sigma**2)))

    #fwhm = 2.
    #sigma = fwhm / 2.3548
    #out += 0.1*np.exp(-((x+20)**2 / (2*sigma**2/2.) + (y+20)**2 / (2*sigma**2)))

    #out += 0.1*np.exp(-((x-20)**2 / (2*0.1**2) + y**2 / (2*0.1**2)))
    #out += 0.5*np.exp(-((x-30)**2 / (2*0.1**2) + y**2 / (2*0.1**2)))
    #out += 0.2*np.exp(-((x)**2 / (2*0.1**2) + (y-5)**2 / (2*0.1**2)))

    #out += gauss2d(1000.0, 0., 0., 0.1, 0.1, x, y)

    add_star(out, x, y, 30., 30., 1000.)
    add_star(out, x, y, 30., -30., 100.)
    add_star(out, x, y, 0., 0., 1000.)

    if not is_reference:
        add_star(out, x, y, 0., 20., 4.0)
        add_star(out, x, y, 0., 10., 2.0)
        add_star(out, x, y, 0., 0., 1.0)
        add_star(out, x, y, 0., -10., 0.5)
        add_star(out, x, y, 0., -20., 0.25)
        add_star(out, x, y, -20., 0., 2.)
        pass

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
shift_x = 0.
shift_y = 0.
for i in range(5):
    psf_data = psf_function(psf_x_grid + shift_x, psf_y_grid + shift_y)
    center_y, center_x = center_of_mass(psf_data)
    add_shift_x = center_x / oversampling - psf_x_max
    add_shift_y = center_y / oversampling - psf_y_max
    print "Shifting by x=%.5f, y=%.5f (data pix)" % (add_shift_x, add_shift_y)
    shift_x += add_shift_x
    shift_y += add_shift_y
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
conv_psf_spline = RectBivariateSpline(psf_y_range, psf_x_range, conv_psf_data)

# Write the convolved PSF to a fits file
conv_psf_hdu = fits.PrimaryHDU(conv_psf_data)
conv_psf_hdulist = fits.HDUList([conv_psf_hdu])
conv_psf_hdulist.writeto('./%s_conv_psf.fits' % (prefix,), clobber=True)

# Generate the data model
num_x = int(0.75*size_x * oversampling)
num_y = int(0.75*size_y * oversampling)
data_x_range = (np.arange(2*num_x+1) - num_x) / oversampling
data_y_range = (np.arange(2*num_y+1) - num_y) / oversampling

data_x_grid, data_y_grid = np.meshgrid(data_x_range, data_y_range)

# data = data_function(data_x_grid, data_y_grid)
# data_spline = RectBivariateSpline(data_x_range, data_y_range, data.T)

for i, dither_data in enumerate(dithers):
    x_offset, y_offset, rotation_angle, is_reference = dither_data
    print "Generating image %d" % i

    # Generate a rotation matrix for the given angle. We generate a clockwise
    # rotation matrix since the specified rotation angle is counter-clockwise
    angle_rad = rotation_angle * np.pi / 180.
    cw_rot_matrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                              [-np.sin(angle_rad), np.cos(angle_rad)]])
    ccw_rot_matrix = np.linalg.inv(cw_rot_matrix)

    rot_x_grid, rot_y_grid = np.einsum(
        'ij, mni -> jmn',
        cw_rot_matrix,
        np.dstack([data_x_grid, data_y_grid])
    )

    # Generate data and convolve with the PSF
    rot_data = data_function(rot_x_grid, rot_y_grid, is_reference)
    psf_applied_data = fftconvolve(
        rot_data,
        conv_psf_data,
        mode='same'
    )

    # Add in stars. This gets the widths exactly right. Important if we are
    # trying to subtract them.
    for star_x, star_y, star_flux in stars:
        rot_star_x, rot_star_y = cw_rot_matrix.dot(np.array([star_x, star_y]))
        star_data = star_flux * conv_psf_spline(
            data_y_range - rot_star_y,
            data_x_range - rot_star_x
        )

        psf_applied_data += star_data

    spline = RectBivariateSpline(data_y_range, data_x_range, psf_applied_data)

    # Sample the data
    sample_x_range = np.arange(-size_x/2. + 1.0 + x_offset, size_x/2. + 1.0 +
                               x_offset)
    sample_y_range = np.arange(-size_y/2. + 1.0 + y_offset, size_y/2. + 1.0 +
                               y_offset)
    dither_data = spline(sample_y_range, sample_x_range)

    # Add noise
    dither_data += np.random.normal(scale=noise_sigma, size=dither_data.shape)

    # Generate the WCS
    ra_offset, dec_offset = ccw_rot_matrix.dot([x_offset, y_offset])
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
    dither_wcs.wcs.pc = ccw_rot_matrix

    # Write the output data to a fits file
    dither_header = dither_wcs.to_header()
    dither_header['ORIENTAT'] = rotation_angle
    dither_header['EXPTIME'] = 100.
    dither_hdu = fits.PrimaryHDU(dither_data, header=dither_header)
    dither_hdulist = fits.HDUList([dither_hdu])
    dither_hdulist.writeto('./%s_dither_%d.fits' % (prefix, i), clobber=True)
