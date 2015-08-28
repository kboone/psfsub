from seechange.data import FitsFile
from scipy.signal import fftconvolve
from astropy.io import fits
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import RectBivariateSpline
import os
import numpy as np
import sep
from scipy.optimize import minimize

basedir = '/home/kboone/optimal_subtraction/psfsub/'
psf_path = basedir + 'f140w_11x00_tinytim_conv.fits'
oversampling = 11
data_basedir = '/home/scpdata05/clustersn/kboone/idcsj/data/'
#data_path = data_basedir + 'SPT0205/vA/F140W/icn111myq_pam.fits'
#data_path = data_basedir + 'SPT0205/vA/F140W/icn111mzq_pam.fits'
#data_path = data_basedir + 'SPT0205/vA/F140W/icn111n7q_pam.fits'
#data_path = data_basedir + 'SPT0205/vC/F140W/icn112aeq_pam.fits'
#data_path = data_basedir + 'SPT0205/vC/F140W/icn112a5q_pam.fits'
#data_path = data_basedir + 'SPT0205/vC/F140W/icn112a6q_pam.fits'
#data_path = data_basedir + 'SPT0205/vE/F140W/icn113ymq_pam.fits'
#data_path = data_basedir + 'SPT0205/vE/F140W/icn113yoq_pam.fits'
data_path = data_basedir + 'SPT0205/vE/F140W/icn113yqq_pam.fits'
#data_path = data_basedir + 'XMM44/v1/F105W/icn164jyq_pam.fits'
#data_path = data_basedir + 'XMM44/v1/F140W/icn164juq_pam.fits'
#data_path = data_basedir + 'XMM44/v1/F140W/icn164jtq_pam.fits'
#data_path = data_basedir + 'XMM44/v1/F140W/icn164k2q_pam.fits'
output_directory = basedir + '/psfs/'
star_list = np.loadtxt(basedir + '/spt0205_stars.txt')


fits_file = FitsFile.open(data_path)
data = fits_file.get_data()
err = fits_file.get_sky_error()
mask = fits_file.get_mask()
mask_float = mask.astype(float)

# Load the starting PSF
psf_hdulist = fits.open(psf_path)
psf_data = psf_hdulist[0].data.byteswap(False).newbyteorder()
center_y, center_x = center_of_mass(psf_data)
y_pix_range = (np.arange(psf_data.shape[0]) - center_y) / oversampling
x_pix_range = (np.arange(psf_data.shape[1]) - center_x) / oversampling
old_spline = RectBivariateSpline(y_pix_range, x_pix_range, psf_data)

# Get a catalog of potential bright stars
bkg = sep.Background(data, mask=mask)
bkg.subfrom(data)
objects = sep.extract(data, 10.0, err=bkg.rms()*(~mask))
objects.sort(order='flux')

# Do some cuts to get a relatively pure sample of stars
cut_dist = 30.
objects = objects[
    (objects['x'] > cut_dist)
    & (objects['x'] < data.shape[1] - cut_dist)
    & (objects['y'] > cut_dist)
    & (objects['y'] < data.shape[0] - cut_dist)
    & (objects['tnpix'] > 10)
]

# Require objects to be in our star list
star_ras = star_list[:, 0]
star_decs = star_list[:, 1]
star_xs, star_ys = fits_file.rd_to_xy(star_ras, star_decs)
cut = (
    (star_xs > cut_dist)
    & (star_xs < data.shape[1] - cut_dist)
    & (star_ys > cut_dist)
    & (star_ys < data.shape[0] - cut_dist)
)
star_ras = star_ras[cut]
star_decs = star_decs[cut]
star_xs = star_xs[cut]
star_ys = star_ys[cut]

good_indices = []

for star_x, star_y in zip(star_xs, star_ys):
    dists = np.sqrt((objects['x'] - star_x)**2 + (objects['y'] - star_y)**2)
    best_match_index = np.argmin(dists)
    if dists[best_match_index] < 5.:
        good_indices.append(best_match_index)

objects = objects[good_indices]

# Calculate approx FWHM and cut on that
sep_obj_psf = sep.extract(psf_data, np.max(psf_data) / 100.)
if len(sep_obj_psf) != 1:
    raise Exception('Multiple objects found in PSF!!!')
sep_obj_psf = sep_obj_psf[0]

# We use this energy fraction with an empirical correction to get the
# approximate standard deviation (calculated on a 2d gaussian distribution)
flux_fraction = 0.7
radius_to_std = 1. / 1.4875
psf_std = sep.flux_radius(
    psf_data,
    sep_obj_psf['x'],
    sep_obj_psf['y'],
    5.*oversampling,
    flux_fraction,
)[0] / oversampling * radius_to_std
obj_stds = sep.flux_radius(
    data,
    objects['x'],
    objects['y'],
    5.*np.ones(len(objects)),
    flux_fraction,
    mask=mask_float
)[0] * radius_to_std

min_over_radius = np.min(obj_stds[obj_stds > psf_std])
#cut = ((obj_stds > psf_std / 2.)
       #& (obj_stds < psf_std * 1.5))
#objects = objects[cut]
#obj_stds = obj_stds[cut]


def gauss2d(amp, center_x, center_y, sigma_x, sigma_y, x, y):
    #norm = amp / (2. * np.pi * sigma_x * sigma_y) / oversampling / oversampling
    norm = 1.
    func = np.exp(-((x-center_x)**2 / (2.*sigma_x**2) + (y-center_y)**2 /
                    (2.*sigma_y**2)))
    return norm * func


def get_psf_spline(sigma_x, sigma_y):
    kernel_range = np.arange(-10., 10.001, 1/float(oversampling))
    kernel_grid_x, kernel_grid_y = np.meshgrid(kernel_range, kernel_range)
    kernel = gauss2d(1., 0., 0., sigma_x, sigma_y, kernel_grid_x,
                     kernel_grid_y)
    kernel /= np.sum(kernel)
    conv_psf_data = fftconvolve(
        psf_data,
        kernel,
        'same'
    )
    return RectBivariateSpline(y_pix_range, x_pix_range, conv_psf_data)

# Estimate the missing variance
psf_var = psf_std * psf_std
data_std = np.median(obj_stds)
data_var = data_std * data_std
if psf_var > data_var:
    raise Exception('Model var > data var! Deconvolve not supported')

correction_x = np.sqrt(data_var - psf_var)
correction_y = np.sqrt(data_var - psf_var)

spline = get_psf_spline(
    correction_x,
    correction_y
)


def single_chisq(data, mask, psf_spline, amp, obj_x, obj_y,
                 launch_ipython=False):
    # Pick a smallish box as I don't really care about getting the tails right.
    box_radius = 10
    x_range_eval = (np.arange(box_radius*2+1) - box_radius +
                    int(np.round(obj_x)))
    y_range_eval = (np.arange(box_radius*2+1) - box_radius +
                    int(np.round(obj_y)))

    x_grid_eval, y_grid_eval = np.meshgrid(x_range_eval, y_range_eval)

    psf_box = amp*psf_spline(y_range_eval - obj_y, x_range_eval - obj_x)
    data_box = data[y_grid_eval, x_grid_eval]
    mask_box = mask[y_grid_eval, x_grid_eval]

    sub = psf_box - data_box

    # Err ~ sqrt(psf_data)
    err2 = sub*sub
    err2[mask_box] = 0.

    denom = np.sum(np.abs(data_box))

    if launch_ipython:
        from IPython import embed; embed()

    return (np.sum(err2), denom)


def total_chisq(amplitudes, xs, ys, correction_x, correction_y,
                do_print=False, launch_ipython=False):
    spline = get_psf_spline(
        correction_x,
        correction_y
    )

    if do_print:
        print "\nPulls:"

    chisq_sum = 0.
    for amp, x, y in zip(amplitudes, xs, ys):
        err2, denom = single_chisq(data, mask, spline, amp, x, y,
                                   launch_ipython=launch_ipython)
        if do_print:
            print x, y, err2 / denom, err2 / denom / denom
        chisq_sum += err2 / denom

    return chisq_sum


def to_minimize(params, do_print=False, launch_ipython=False):
    correction_x = params[0]
    correction_y = params[1]
    num_objs = (len(params) - 2) / 3
    fluxes = params[2:2+num_objs]
    xs = params[2+num_objs:2+2*num_objs]
    ys = params[2+2*num_objs:2+3*num_objs]
    chisq_sum = total_chisq(fluxes, xs, ys, correction_x, correction_y,
                            do_print=do_print, launch_ipython=launch_ipython)
    #print correction_x, correction_y, chisq_sum
    return chisq_sum

all_cor_x = []
all_cor_y = []

# Calculate initial chisq fits for all of the data
for i, obj in enumerate(objects):
    print "%2d/%d" % (i+1, len(objects)),
    params = [correction_x, correction_y, obj['flux'], obj['x'], obj['y']]
    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (obj['x'] - 0.2, obj['x'] + 0.2),
        (obj['y'] - 0.2, obj['y'] + 0.2),
    ]
    res = minimize(to_minimize, params, bounds=bounds)
    fit_cor_x, fit_cor_y, fit_flux, fit_x, fit_y = res['x']

    fit_cor_x = np.abs(fit_cor_x)
    fit_cor_y = np.abs(fit_cor_y)

    status = (
        "success = %s, cor = (%.4f, %.4f), flux ratio = % 6.3f, "
        "shift = (%5.3f, %5.3f)" % (
            res['success'],
            fit_cor_x,
            fit_cor_y,
            fit_flux / obj['flux'],
            fit_x - obj['x'],
            fit_y - obj['y']
        )
    )
    print status

    all_cor_x.append(fit_cor_x)
    all_cor_y.append(fit_cor_y)

    #from IPython import embed; embed()

print "Final results: cor = (%.4f, %.4f)" % (np.median(all_cor_x),
                                             np.median(all_cor_y))

# Output final psf
basename = os.path.splitext(os.path.basename(data_path))[0]
out_path = output_directory + "/" + basename + "_psf.fits"

spline = get_psf_spline(np.median(all_cor_x), np.median(all_cor_y))

shape_y, shape_x = psf_data.shape
y_out_range = (np.arange(shape_y) - (shape_y-1)/2) / float(oversampling)
x_out_range = (np.arange(shape_x) - (shape_x-1)/2) / float(oversampling)

psf_out = spline(y_out_range, x_out_range)

# Write the PSF to a fits file
psf_hdu = fits.PrimaryHDU(psf_out)
psf_hdulist = fits.HDUList([psf_hdu])
psf_hdulist.writeto(out_path, clobber=True)


#import sys
#sys.exit()

#params = [correction_x, correction_y]
#params.extend(objects['flux'])
#params.extend(objects['x'])
#params.extend(objects['y'])
#bounds = [(None, None), (None, None)]
#bounds.extend([(None, None)]*len(objects))
#bounds.extend(zip(objects['x'] - 0.2, objects['x'] + 0.2))
#bounds.extend(zip(objects['y'] - 0.2, objects['y'] + 0.2))

##res = minimize(to_minimize, params, bounds=bounds)
##for obj in objects[::-1]:
    ##print obj['x'], obj['y']
    ##chisq(data, mask, spline, obj['flux'], obj['x'], obj['y'])
#out_values = res['x']
#out_cor_x = out_values[0]
#out_cor_y = out_values[1]
#out_amplitudes = out_values[2:]

#to_minimize(out_values, do_print=True)

##total_chisq(out_amplitudes, objects['x'], objects['y'], out_cor_x, out_cor_y,
            ##do_print=True, launch_ipython=True)

#from IPython import embed; embed()
