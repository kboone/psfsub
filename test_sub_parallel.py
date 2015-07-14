#!/usr/bin/env python

import sys
from psfsub.data import Image
import numpy as np
from seechange.data import FitsFile
from matplotlib import pyplot as plt

base = '/home/scpdata05/clustersn/kboone/idcsj/data/'

ref_images = [
    Image(base + 'SPARCS-J003550-431210/v1/F140W/icn199axq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    Image(base + 'SPARCS-J003550-431210/v1/F140W/icn199ayq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    Image(base + 'SPARCS-J003550-431210/v1/F140W/icn199b6q_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11)
]
new_images = [
    Image(base + 'SPARCS-J003550-431210/v2_fakes/F140W/icn173yzq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    Image(base + 'SPARCS-J003550-431210/v2_fakes/F140W/icn173z1q_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    Image(base + 'SPARCS-J003550-431210/v2_fakes/F140W/icn173z3q_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11)
]

scale_image = FitsFile.open(
    base + 'SPARCS-J003550-431210/sets/SPARCS-J003550-431210_v1_F140W.fits',
    readonly=True
)

#base = '/home/scpdata05/clustersn/kboone/idcsj/data/'

#ref_images = [
    #Image(base + 'SPT0205/vC/F140W/icn112a5q_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPT0205/vC/F140W/icn112a6q_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPT0205/vC/F140W/icn112aeq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11)
#]
#new_images = [
    #Image(base + 'SPT0205/vE_fakes/F140W/icn113ymq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPT0205/vE_fakes/F140W/icn113yoq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPT0205/vE_fakes/F140W/icn113yqq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits', 11)
#]

#scale_image = FitsFile.open(
    #base + 'SPT0205/sets/SPT0205_v1_F140W.fits',
    #readonly=True
#)

#ref_images = [
    #Image(base + 'SPARCSJ0330/v1/F105W/icn135ffq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPARCSJ0330/v1/F105W/icn135fhq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPARCSJ0330/v1/F105W/icn135fmq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11)
#]
#new_images = [
    #Image(base + 'SPARCSJ0330/v2_psfsub_test/F105W/icn136aaq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPARCSJ0330/v2_psfsub_test/F105W/icn136adq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image(base + 'SPARCSJ0330/v2_psfsub_test/F105W/icn136afq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11)
#]

#scale_image = FitsFile.open(
    #base + 'SPARCSJ0330/sets/SPARCSJ0330_v1_F105W.fits',
    #readonly=True
#)

naxis1 = scale_image.naxis1[0]
naxis2 = scale_image.naxis2[0]

center_ra, center_dec = scale_image.get_wcs().wcs.crval
cos_dec = np.cos(center_dec*np.pi/180.)

new_psf = new_images[0].psfs[0]
ref_psf = ref_images[0].psfs[0]
gg_spline = new_psf.get_convolution_spline(new_psf)
gh_spline = new_psf.get_convolution_spline(ref_psf)
hh_spline = ref_psf.get_convolution_spline(ref_psf)


def do_subtraction(ra, dec, radius=0.2, amp=1.0, f=0.2):
    #print "Doing subtraction for (%f, %f)" % (ra, dec)
    ref_data = np.hstack([i.find_near(ra, dec, radius) for i in ref_images])
    new_data = np.hstack([i.find_near(ra, dec, radius) for i in new_images])

    ref_vals, ref_errs, ref_psfs, ref_ras, ref_decs = ref_data
    new_vals, new_errs, new_psfs, new_ras, new_decs = new_data

    if len(ref_vals) == 0 or len(new_vals) == 0:
        return 0

    # TODO: Don't use data like this when estimating f. It's very wrong.
    f = np.median(ref_vals)
    #f_gg = np.outer(new_vals, new_vals)
    #f_gh = np.outer(new_vals, ref_vals)
    #f_hh = np.outer(ref_vals, ref_vals)

    new_n = np.diag(new_errs**2)
    ref_n = np.diag(ref_errs**2)

    #ref_count = len(ref_vals)
    #new_count = len(new_vals)

    #print "Number of indices to use: %d ref, %d new" % (ref_count, new_count)

    # TODO: deal with multiple psfs
    #new_psf = new_psfs[0]
    #ref_psf = ref_psfs[0]

    #print "Generating system matrices"

    # TODO: Check that the order of my differences is right!

    diff_x = np.subtract.outer(new_ras, new_ras) * cos_dec / 0.13 * 3600.
    diff_y = np.subtract.outer(new_decs, new_decs) / 0.13 * 3600.
    gg = gg_spline.ev(diff_x, diff_y)

    diff_x = np.subtract.outer(new_ras, ref_ras) * cos_dec / 0.13 * 3600.
    diff_y = np.subtract.outer(new_decs, ref_decs) / 0.13 * 3600.
    gh = gh_spline.ev(diff_x, diff_y)

    diff_x = np.subtract.outer(ref_ras, ref_ras) * cos_dec / 0.13 * 3600.
    diff_y = np.subtract.outer(ref_decs, ref_decs) / 0.13 * 3600.
    hh = hh_spline.ev(diff_x, diff_y)

    diff_x = (ra - new_ras) * cos_dec / 0.13 * 3600.
    diff_y = (dec - new_decs) / 0.13 * 3600.
    gsn = new_psf.psf_spline.ev(diff_x, diff_y)

    #print "Calculating ABCD matrices"
    a = gg*f**2 + amp**2 * np.outer(gsn, gsn) + new_n
    b = -2. * amp**2 * gsn
    c = -2. * gh*f**2
    d = hh*f**2 + ref_n

    #print "Calculating S and T matrices"

    t = - np.linalg.inv(2*a - 1/2.*c.dot(np.linalg.inv(d).dot(c.T))).dot(b)
    s = -1/2. * np.linalg.inv(d).dot(c.T.dot(t).T)

    #print "Sum of t,s: ", np.sum(t), np.sum(s)

    #print "Median values", np.median(new_vals), np.median(ref_vals)

    #print t.dot(new_vals)
    #print s.dot(ref_vals)

    #print t.dot(new_vals) - s.dot(ref_vals)

    #print np.sum(t), np.sum(s)
    #print np.median(new_vals), np.median(ref_vals)
    #print np.mean(new_vals), np.mean(ref_vals)
    #print np.sum(t) * np.median(new_vals) - np.sum(s) * np.median(ref_vals)

    return t.dot(new_vals) - s.dot(ref_vals)


def do_grid(x_range, y_range, **kwargs):
    """Generate a grid of subtraction around some ra and dec"""

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    ra_grid, dec_grid = scale_image.xy_to_rd(x_grid, y_grid)

    out_grid = np.zeros(ra_grid.shape)

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            out_grid[j, i] = do_subtraction(ra_grid[j, i], dec_grid[j, i],
                                            **kwargs)

    return out_grid


def showim(data):
    """Nice imshow that does things right"""
    plt.ion()
    plt.figure()
    scale = np.median(np.abs(data))
    plt.imshow(data, origin='lower', cmap='gray', interpolation='none',
               vmin=-scale*10., vmax=scale*15.)
    plt.colorbar()

if __name__ == "__main__":
    args = sys.argv

    try:
        x_tile = int(args[1])
        num_tiles_x = int(args[2])
        y_tile = int(args[3])
        num_tiles_y = int(args[4])
    except:
        print "Invalid arguments, format is:"
        print "%s [x_tile] [num_tiles_x] [y_tile] [num_tiles_y]" % args[0]
        sys.exit(1)

    num_per_tile_x = naxis1 / num_tiles_x
    start_x = x_tile * num_per_tile_x + 1
    end_x = (x_tile + 1) * num_per_tile_x + 1
    if end_x > naxis1:
        end_x = naxis1

    num_per_tile_y = naxis2 / num_tiles_y
    start_y = y_tile * num_per_tile_y + 1
    end_y = (y_tile + 1) * num_per_tile_y + 1
    if end_y > naxis2:
        end_y = naxis2

    x_range = np.arange(start_x, end_x)
    y_range = np.arange(start_y, end_y)

    data = do_grid(x_range, y_range)

    np.save('./grid_%d_%d.npy' % (x_tile, y_tile), data)
