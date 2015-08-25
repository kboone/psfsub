from psfsub import Subtractor
import cProfile
import pstats

base = '/home/scpdata05/clustersn/kboone/idcsj/data/'

psf = '/home/scpdata05/wfc3/PSF_iso/f140w_11x00_convolved_norm.fits'


s = Subtractor()

s.add_reference(base + 'SPT0205/vA/F140W/icn111myq_pam.fits', psf, 11)
s.add_reference(base + 'SPT0205/vA/F140W/icn111mzq_pam.fits', psf, 11)
s.add_reference(base + 'SPT0205/vA/F140W/icn111n7q_pam.fits', psf, 11)
s.add_reference(base + 'SPT0205/vC/F140W/icn112a5q_pam.fits', psf, 11)
s.add_reference(base + 'SPT0205/vC/F140W/icn112a6q_pam.fits', psf, 11)
s.add_reference(base + 'SPT0205/vC/F140W/icn112aeq_pam.fits', psf, 11)
s.add_new(base + 'SPT0205/vE_fakes/F140W/icn113ymq_pam.fits', psf, 11)
s.add_new(base + 'SPT0205/vE_fakes/F140W/icn113yoq_pam.fits', psf, 11)
s.add_new(base + 'SPT0205/vE_fakes/F140W/icn113yqq_pam.fits', psf, 11)
s.read_output_coordinates(base + 'SPT0205/sets/SPT0205_v1_F140W.fits')

#s.add_reference(base + 'SPT2040/v1/F140W/icn128g3q_pam.fits', psf, 11)
#s.add_reference(base + 'SPT2040/v1/F140W/icn128g4q_pam.fits', psf, 11)
#s.add_reference(base + 'SPT2040/v1/F140W/icn128gcq_pam.fits', psf, 11)
#s.add_reference(base + 'SPT2040/v2/F140W/icn129j8q_pam.fits', psf, 11)
#s.add_reference(base + 'SPT2040/v2/F140W/icn129jaq_pam.fits', psf, 11)
#s.add_reference(base + 'SPT2040/v2/F140W/icn129jcq_pam.fits', psf, 11)
#s.add_new(base + 'SPT2040/v3_fakes/F140W/icn130mrq_pam.fits', psf, 11)
#s.add_new(base + 'SPT2040/v3_fakes/F140W/icn130mtq_pam.fits', psf, 11)
#s.add_new(base + 'SPT2040/v3_fakes/F140W/icn130mvq_pam.fits', psf, 11)
#s.read_output_coordinates(base + 'SPT2040/sets/SPT2040_v1_F140W.fits')

#s.add_new(base + 'SPARCS-J1049/v2/F160W/icn184b1q_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v2/F160W/icn184b2q_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v2/F160W/icn184b4q_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v3/F160W/icn185v0q_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v3/F160W/icn185v1q_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v3/F160W/icn185v9q_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v4/F160W/icn186f9q_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v4/F160W/icn186fgq_pam.fits', psf, 11)
#s.add_new(base + 'SPARCS-J1049/v4/F160W/icn186fiq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v5/F160W/iclq01duq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v5/F160W/iclq01dvq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v5/F160W/iclq01dxq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v6/F160W/iclq02i9q_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v6/F160W/iclq02iaq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v6/F160W/iclq02icq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v7/F160W/icn194joq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v7/F160W/icn194jpq_pam.fits', psf, 11)
#s.add_reference(base + 'SPARCS-J1049/v7/F160W/icn194jrq_pam.fits', psf, 11)
#s.read_output_coordinates(base + 'SPARCS-J1049/sets/SPARCS-J1049_v1_F160W.fits')

#s.add_new(base + 'SPT0205/vD/F140W/icn10dfmq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vD/F140W/icn10dfoq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vE/F140W/icn113ymq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vE/F140W/icn113yoq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vE/F140W/icn113yqq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vF/F140W/icn10dfmq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vF/F140W/icn10dfoq_pam.fits', psf, 11)


s.output_naxis1 = 400
s.output_naxis2 = 400

#s.output_center_ra = 31.444359
#s.output_center_dec = -58.485358

#s.output_center_ra = 162.366633
#s.output_center_dec = 56.671097

#s.output_center_ra = 31.431610
#s.output_center_dec = -58.487506

#s.output_center_ra = 31.414949
#s.output_center_dec = -58.493038

#s.output_center_ra = 31.433153
#s.output_center_dec = -58.486871

#s.output_center_ra = 31.452069
#s.output_center_dec = -58.486277

#s.output_center_ra = 310.237309
#s.output_center_dec = -44.857808

#s.output_center_ra = 31.452102
#s.output_center_dec = -58.489747

#s.output_center_ra = 31.445539
#s.output_center_dec = -58.482944

#s.output_center_ra = 31.45050
#s.output_center_dec = -58.49075

#s.output_center_ra = 31.459094
#s.output_center_dec = -58.500716


#s.output_pixel_scale = 0.05

s.load_data()


def do_profile():
    cProfile.run('s.do_subtraction()', 'test_output.stats')
    stats = pstats.Stats('./test_output.stats')
    stats.strip_dirs().sort_stats('tottime').print_stats(50)

sub_grid, err_grid = s.do_subtraction()
#do_profile()


#from IPython import embed; embed()

import sys; sys.exit()








s2 = Subtractor()

psf = '/home/scpdata05/wfc3/PSF_iso/f105w_11x00_convolved_norm.fits'

s2.add_reference(base + 'SPT0205/vA/F105W/icn111n1q_pam.fits', psf, 11)
s2.add_reference(base + 'SPT0205/vA/F105W/icn111n3q_pam.fits', psf, 11)
s2.add_reference(base + 'SPT0205/vA/F105W/icn111n5q_pam.fits', psf, 11)
s2.add_reference(base + 'SPT0205/vC/F105W/icn112a8q_pam.fits', psf, 11)
s2.add_reference(base + 'SPT0205/vC/F105W/icn112aaq_pam.fits', psf, 11)
s2.add_reference(base + 'SPT0205/vC/F105W/icn112acq_pam.fits', psf, 11)
s2.add_new(base + 'SPT0205/vE_fakes/F105W/icn113yhq_pam.fits', psf, 11)
s2.add_new(base + 'SPT0205/vE_fakes/F105W/icn113yiq_pam.fits', psf, 11)
s2.add_new(base + 'SPT0205/vE_fakes/F105W/icn113ykq_pam.fits', psf, 11)

s2.read_output_coordinates(base + 'SPT0205/sets/SPT0205_v1_F140W.fits')

s2.output_naxis1 = 100
s2.output_naxis2 = 100

s2.output_center_ra = 31.414949
s2.output_center_dec = -58.493038

#s2.output_pixel_scale = 0.04

s2.load_data()

sub_grid2, err_grid2 = s2.do_subtraction()

out = (sub_grid / err_grid / err_grid**2 + sub_grid2 / err_grid2 /
       err_grid2**2) / (1 / err_grid**2 + 1 / err_grid2**2)

#from IPython import embed; embed()
