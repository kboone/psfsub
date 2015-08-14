from psfsub import Subtractor
import cProfile
import pstats

base = '/home/kboone/optimal_subtraction/psfsub/test_bob_1/'

psf = base + 'gen_psf.fits'


s = Subtractor()

s.add_reference(base + 'gen_dither_1.fits', psf, 11)
s.add_reference(base + 'gen_dither_2.fits', psf, 11)
s.add_reference(base + 'gen_dither_3.fits', psf, 11)
s.add_reference(base + 'gen_dither_4.fits', psf, 11)
s.add_reference(base + 'gen_dither_5.fits', psf, 11)
s.add_reference(base + 'gen_dither_6.fits', psf, 11)
s.add_new(base + 'gen_dither_7.fits', psf, 11)
s.add_new(base + 'gen_dither_8.fits', psf, 11)
s.add_new(base + 'gen_dither_9.fits', psf, 11)
#s.add_new(base + 'SPT0205/vD/F140W/icn10dfmq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vD/F140W/icn10dfoq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vE/F140W/icn113ymq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vE/F140W/icn113yoq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vE/F140W/icn113yqq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vF/F140W/icn10dfmq_pam.fits', psf, 11)
#s.add_new(base + 'SPT0205/vF/F140W/icn10dfoq_pam.fits', psf, 11)

s.read_output_coordinates(base + 'gen_dither_1.fits')

s.output_naxis1 = 150
s.output_naxis2 = 150
s.output_pixel_scale = 0.08

#s.output_center_ra = 31.414949
#s.output_center_dec = -58.493038

#s.output_center_ra = 31.433153
#s.output_center_dec = -58.486871

#s.output_center_ra = 31.452102
#s.output_center_dec = -58.489747

#s.output_center_ra = 31.445539
#s.output_center_dec = -58.482944

#s.output_center_ra = 31.45050
#s.output_center_dec = -58.49075

#s.output_center_ra = 31.459094
#s.output_center_dec = -58.500716


s.load_data()


def do_profile():
    cProfile.run('s.do_subtraction()', 'test_output.stats')
    stats = pstats.Stats('./test_output.stats')
    stats.strip_dirs().sort_stats('tottime').print_stats(50)

sub_grid, err_grid = s.do_subtraction()
#do_profile()


#from IPython import embed; embed()
