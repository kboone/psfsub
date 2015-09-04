import numpy as np
from astropy.io import fits
from scipy.signal import resample, fftconvolve
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import RectBivariateSpline

basedir = '/home/kboone/optimal_subtraction/psfsub/'
psf_path = basedir + 'tinytim_psf_f140w_bb5500.fits'
in_pixel_scale = 0.039242
output_path = basedir + 'f140w_11x00_tinytim_noconv.fits'
pixel_scale_x = 0.13543
pixel_scale_y = 0.12096
oversampling = 11

data = fits.open(psf_path)[0].data
data = data.byteswap(False).newbyteorder()

initial_oversampling = 5
data = resample(data, data.shape[0]*initial_oversampling, axis=0)
data = resample(data, data.shape[1]*initial_oversampling, axis=1)

data_pixel_scale = in_pixel_scale / initial_oversampling
center_y, center_x = center_of_mass(data)
x_pix_range = (np.arange(data.shape[0]) - center_x) * data_pixel_scale
y_pix_range = (np.arange(data.shape[0]) - center_y) * data_pixel_scale
x_pix_grid, y_pix_grid = np.meshgrid(x_pix_range, y_pix_range)


def gauss2d(amp, center_x, center_y, sigma_x, sigma_y, x, y):
    norm = amp / (2. * np.pi * sigma_x * sigma_y) / oversampling / oversampling
    func = np.exp(-((x-center_x)**2 / (2.*sigma_x**2) + (y-center_y)**2 /
                    (2.*sigma_y**2)))
    return norm * func

# Convolve with the inter-pixel capacitance
# NOTE: This is done later now.
# Note: the PSF will get shifted here if the input is off center. Whatever,
# it's fine.
# Note: To tune this, uncomment the following lines and tweak until you get the
# kernel from the WFC3 manual as the output.
# data[:, :] = 0.
# data[data.shape[0]/2, data.shape[1]/2] = 1.
# (add the next lines to the end)
# print (conv_data[165, 165], conv_data[165, 176],
# conv_data[176, 165], conv_data[176, 176])

#ratio = pixel_scale_y / pixel_scale_x
#sigma_x = 0.03925
#sigma_y = ratio * sigma_x
#capacitance = gauss2d(1.0, 0., 0., sigma_x, sigma_y, x_pix_grid, y_pix_grid)
#capacitance /= np.sum(capacitance)
#cap_data = fftconvolve(data, capacitance, mode='same')
cap_data = data

# Get a spline to sample properly on the output grid.
center_y, center_x = center_of_mass(cap_data)
x_pix_range = (np.arange(cap_data.shape[0]) - center_x) * data_pixel_scale
y_pix_range = (np.arange(cap_data.shape[0]) - center_y) * data_pixel_scale
spline = RectBivariateSpline(y_pix_range, x_pix_range, cap_data)
x_range = np.arange(-15.*pixel_scale_x, 15.0001*pixel_scale_x, pixel_scale_x /
                    oversampling)
y_range = np.arange(-15.*pixel_scale_y, 15.0001*pixel_scale_y, pixel_scale_y /
                    oversampling)
x_grid, y_grid = np.meshgrid(x_range, y_range)
out_data = spline.ev(y_grid, x_grid)
out_data /= np.sum(out_data)

# Convolve with the interpixel capacitance kernel
kernel = np.zeros((2*oversampling+1, 2*oversampling+1))
corner_val = 0.0007
side_val = 0.025
center_val = 0.897
kernel[0, 0] = corner_val
kernel[0, 2*oversampling] = corner_val
kernel[2*oversampling, 0] = corner_val
kernel[2*oversampling, 2*oversampling] = corner_val
kernel[oversampling, 0] = side_val
kernel[0, oversampling] = side_val
kernel[2*oversampling, oversampling] = side_val
kernel[oversampling, 2*oversampling] = side_val
kernel[oversampling, oversampling] = center_val
pix_cap_data = fftconvolve(
    out_data,
    kernel,
    mode='same'
)
pix_cap_data /= np.sum(pix_cap_data)

print "SKIPPING INTERPIX CAP"
pix_cap_data = out_data


# Convolve with the pixel shape
conv_data = fftconvolve(
    pix_cap_data,
    np.ones((oversampling, oversampling)),
    mode='same'
)

# Save the PSF.
psf_hdu = fits.PrimaryHDU(conv_data)
psf_hdulist = fits.HDUList([psf_hdu])
psf_hdulist.writeto(output_path)
