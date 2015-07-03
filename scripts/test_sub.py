#!/usr/bin/env python

from psfsub.data import Image
import numpy as np

ref_images = [
    Image('/home/scpdata05/clustersn/kboone/idcsj/data/SPARCSJ0330/v1/F105W/icn135ffq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    Image('/home/scpdata05/clustersn/kboone/idcsj/data/SPARCSJ0330/v1/F105W/icn135fhq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    Image('/home/scpdata05/clustersn/kboone/idcsj/data/SPARCSJ0330/v1/F105W/icn135fmq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11)
]
new_images = [
    Image('/home/scpdata05/clustersn/kboone/idcsj/data/SPARCSJ0330/v2/F105W/icn136aaq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    Image('/home/scpdata05/clustersn/kboone/idcsj/data/SPARCSJ0330/v2/F105W/icn136adq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    Image('/home/scpdata05/clustersn/kboone/idcsj/data/SPARCSJ0330/v2/F105W/icn136afq_pam.fits',
          '/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11)
]

#ref_images = [
    #Image('/home/scpdata05/clustersn/kboone/copy_candidate/test_db_16/MOO-1014/v1/F105W/icn145ahq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image('/home/scpdata05/clustersn/kboone/copy_candidate/test_db_16/MOO-1014/v1/F105W/icn145ajq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image('/home/scpdata05/clustersn/kboone/copy_candidate/test_db_16/MOO-1014/v1/F105W/icn145alq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11)
#]
#new_images = [
    #Image('/home/scpdata05/clustersn/kboone/copy_candidate/test_db_16/MOO-1014/v2_fakes/F105W/icn146e9q_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image('/home/scpdata05/clustersn/kboone/copy_candidate/test_db_16/MOO-1014/v2_fakes/F105W/icn146ebq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11),
    #Image('/home/scpdata05/clustersn/kboone/copy_candidate/test_db_16/MOO-1014/v2_fakes/F105W/icn146edq_pam.fits',
          #'/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits', 11)
#]

center_ra, center_dec = new_images[0].wcs.wcs.crval
cos_dec = np.cos(center_dec*np.pi/180.)

new_psf = new_images[0].psfs[0]
ref_psf = ref_images[0].psfs[0]
gg_spline = new_psf.get_convolution_spline(new_psf)
gh_spline = new_psf.get_convolution_spline(ref_psf)
hh_spline = ref_psf.get_convolution_spline(ref_psf)



def do_subtraction(ra, dec):
    print "Doing subtraction for (%f, %f)" % (ra, dec)
    ref_data = np.hstack([i.find_near(ra, dec, 0.5) for i in ref_images])
    new_data = np.hstack([i.find_near(ra, dec, 0.5) for i in new_images])

    ref_vals, ref_errs, ref_psfs, ref_ras, ref_decs = ref_data
    new_vals, new_errs, new_psfs, new_ras, new_decs = new_data

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

    amp = 0.5
    f = 0.5
    new_n = np.diag(new_errs**2)
    ref_n = np.diag(ref_errs**2)
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

    return t.dot(new_vals) - s.dot(ref_vals)


def do_grid(ra, dec, num_side=10, pixel_sep=0.1):
    """Generate a grid of subtraction around some ra and dec"""
    out_grid = np.zeros((num_side*2+1, num_side*2+1))

    ra_range = (ra + np.arange(-num_side, num_side+1) * pixel_sep / 3600. /
                cos_dec)
    dec_range = (dec + np.arange(num_side, -num_side-1, -1) * pixel_sep /
                 3600.)

    for i, ra_val in enumerate(ra_range):
        for j, dec_val in enumerate(dec_range):
            out_grid[j, i] = do_subtraction(ra_val, dec_val)

    return out_grid


if __name__ == "__main__":
    do_subtraction(center_ra, center_dec)
