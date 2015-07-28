from psfsub import Subtractor
import cProfile

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

s.output_naxis1 = 100
s.output_naxis2 = 100

s.load_data()

cProfile.run('s.do_subtraction()', 'test_output_grid4')
