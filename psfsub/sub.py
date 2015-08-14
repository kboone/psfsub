import numpy as np
import threading
import multiprocessing as mp
from multiprocessing import sharedctypes
import logging
import time
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from data import Image

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Subtractor(object):
    """Class to perform subtractions"""
    def __init__(self):
        # Data
        self.ref_images = []
        self.new_images = []

        self.ref_vals = None
        self.ref_errs = None
        self.ref_psfs = None
        self.ref_ras = None
        self.ref_decs = None

        self.new_vals = None
        self.new_errs = None
        self.new_psfs = None
        self.new_ras = None
        self.new_decs = None

        # Output information
        self.output_naxis1 = None
        self.output_naxis2 = None
        self.output_center_ra = None
        self.output_center_dec = None
        self.output_pixel_scale = None

    def add_reference(self, path, psf_path, psf_oversampling):
        """Add a reference file"""
        image = Image(path, psf_path, psf_oversampling)
        self._add_image(image, is_reference=True)

    def add_new(self, path, psf_path, psf_oversampling):
        """Add a new file"""
        image = Image(path, psf_path, psf_oversampling)
        self._add_image(image, is_reference=False)

    def _add_image(self, image, is_reference):
        """Add an image to be processed"""
        if is_reference:
            self.ref_images.append(image)
        else:
            self.new_images.append(image)

    def read_output_coordinates(self, path, extension='SCI'):
        """Read the output coordinates from a fits file.

        This will copy the WCS parameters from that fits file so that the grids
        are the same. The output should have a single science extension, be
        aligned north up and not have distortion.
        """
        hdulist = fits.open(path, mode='readonly')

        sci_header = hdulist[extension].header

        self.output_naxis1 = sci_header['NAXIS1']
        self.output_naxis2 = sci_header['NAXIS2']

        wcs = WCS(sci_header, fobj=hdulist)
        self.output_center_ra, self.output_center_dec = wcs.wcs.crval

        cd_matrix = wcs.pixel_scale_matrix
        pixel_scale = 3600.*np.sqrt(cd_matrix[0, 0]**2 + cd_matrix[0, 1]**2)
        self.output_pixel_scale = pixel_scale

    def load_data(self):
        """Load all of the data, and get it ready for processing.

        This should be called after all of the relevant images have been added
        with add_reference and add_new
        """
        print "Combining all data"

        # Set up the output WCS object
        output_wcs = WCS(naxis=2)
        output_wcs.wcs.crpix = [self.output_naxis1 / 2.,
                                self.output_naxis2 / 2.]
        output_wcs.wcs.crval = [self.output_center_ra, self.output_center_dec]
        output_wcs.wcs.cdelt = np.array([-self.output_pixel_scale,
                                         self.output_pixel_scale]) / 3600.
        output_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        self.output_wcs = output_wcs

        # Load all of the data.
        # TODO: We use double the memory that we need to currently. This should
        # be optimized to query all of the files for how many points they have,
        # and then load them one by one into preallocated arrays.
        self.ref_vals = np.hstack([i.vals for i in self.ref_images])
        self.ref_errs = np.hstack([i.errs for i in self.ref_images])
        self.ref_psfs = np.hstack([i.psfs for i in self.ref_images])
        self.ref_ras = np.hstack([i.ras for i in self.ref_images])
        self.ref_decs = np.hstack([i.decs for i in self.ref_images])

        self.new_vals = np.hstack([i.vals for i in self.new_images])
        self.new_errs = np.hstack([i.errs for i in self.new_images])
        self.new_psfs = np.hstack([i.psfs for i in self.new_images])
        self.new_ras = np.hstack([i.ras for i in self.new_images])
        self.new_decs = np.hstack([i.decs for i in self.new_images])

        # Convert the ras and decs to arcseconds. This will make all of the
        # numbers look really weird internally, but means that we don't have to
        # scale every time when doing comparisons.
        self.ref_ras *= np.cos(self.output_center_dec * np.pi / 180.) * 3600.
        self.new_ras *= np.cos(self.output_center_dec * np.pi / 180.) * 3600.
        self.ref_decs *= 3600.
        self.new_decs *= 3600.

        # Generate KDTrees.
        # TODO: Try to use a single KDTree for both? Profile this, and if the
        # KDTree is limiting then that might be a good idea.
        print "Generating KDTrees"
        self.ref_kdtree = cKDTree(zip(self.ref_ras, self.ref_decs))
        self.new_kdtree = cKDTree(zip(self.new_ras, self.new_decs))

        print "Done loading data"

    def do_subtraction(self, num_cores=24, num_tiles_x=1, num_tiles_y=1,
                       x_tile=0, y_tile=0, radius=0.3):
        """Do the subtraction.

        This can be done with tiles to easily split the work between different
        machines. The default settings will do the full image.
        """

        # Figure out the pixel indices that we need to run on. Note that the
        # indices are 1 based.
        num_per_tile_x = int(np.ceil(self.output_naxis1 / float(num_tiles_x)))
        start_x = x_tile * num_per_tile_x + 1
        end_x = (x_tile + 1) * num_per_tile_x + 1
        if end_x > self.output_naxis1:
            end_x = self.output_naxis1 + 1

        num_per_tile_y = int(np.ceil(self.output_naxis2 / float(num_tiles_y)))
        start_y = y_tile * num_per_tile_y + 1
        end_y = (y_tile + 1) * num_per_tile_y + 1
        if end_y > self.output_naxis2:
            end_y = self.output_naxis2 + 1

        x_range = np.arange(start_x, end_x)
        y_range = np.arange(start_y, end_y)

        x_grid, y_grid = np.meshgrid(x_range, y_range)
        ra_grid, dec_grid = self.output_wcs.all_pix2world(x_grid, y_grid, 1)

        # Convert the ras and decs to arcseconds.
        ra_grid *= np.cos(self.output_center_dec * np.pi / 180.) * 3600.
        dec_grid *= 3600.

        # Split up the work on all of the cores that we have available
        # Note: using threads kind of sucks because the spline interpolation
        # which is our time limiting factor doesn't disable the GIL. Only
        # processes actually work for speedup now.
        use_threads = False
        if num_cores > 1:
            if use_threads:
                # Use multithreading
                sub_grid = np.zeros(ra_grid.shape)
                err_grid = np.zeros(ra_grid.shape)

                threads = []
                for i in range(num_cores):
                    num_per_core = int(np.ceil(len(y_range) /
                                               float(num_cores)))
                    core_start_index = i * num_per_core
                    core_end_index = (i + 1) * num_per_core
                    if core_end_index > len(y_range):
                        core_end_index = len(y_range)

                    thread = threading.Thread(
                        target=self._do_thread_subtraction,
                        args=(sub_grid, sub_grid.shape, err_grid,
                              err_grid.shape, ra_grid, dec_grid,
                              core_start_index, core_end_index, radius)
                    )
                    threads.append(thread)
                    thread.start()

                # Wait for our threads to end
                for thread in threads:
                    thread.join()
            else:
                # Use multiprocessing
                processes = []

                sub_grid_shared = mp.Array('d', np.product(ra_grid.shape))
                err_grid_shared = mp.Array('d', np.product(ra_grid.shape))

                sub_grid = np.frombuffer(sub_grid_shared.get_obj())
                sub_grid = sub_grid.reshape(ra_grid.shape)
                err_grid = np.frombuffer(err_grid_shared.get_obj())
                err_grid = err_grid.reshape(ra_grid.shape)

                for i in range(num_cores):
                    num_per_core = int(np.ceil(len(y_range) /
                                               float(num_cores)))
                    core_start_index = i * num_per_core
                    core_end_index = (i + 1) * num_per_core
                    if core_end_index > len(y_range):
                        core_end_index = len(y_range)

                    process = mp.Process(
                        target=self.__class__._do_thread_subtraction,
                        args=(self, sub_grid_shared, sub_grid.shape,
                              err_grid_shared, err_grid.shape, ra_grid,
                              dec_grid, core_start_index, core_end_index,
                              radius)
                    )
                    processes.append(process)
                    process.start()

                # Wait for our threads to end
                for process in processes:
                    process.join()
        else:
            # Single core, just call it directly
            sub_grid = np.zeros(ra_grid.shape)
            err_grid = np.zeros(ra_grid.shape)

            self._do_thread_subtraction(
                sub_grid, sub_grid.shape, err_grid, err_grid.shape, ra_grid,
                dec_grid, 0, len(y_range), radius
            )

        np.save('./grid_sub_%d_%d.npy' % (x_tile, y_tile), sub_grid)
        np.save('./grid_err_%d_%d.npy' % (x_tile, y_tile), err_grid)

        return sub_grid, err_grid

    __psf_convolution_cache = {}

    def _get_psf_convolution(self, psf1, psf2):
        """Get the convolution of two PSFs.

        We use a cache so that calling this multiple times actually returns the
        same object.
        """
        # We optimize and return copies of the same object. This is done by
        # maintaining a cache of opened objects, and checking if it is in
        # there.
        index = (psf1, psf2)
        if index in self.__psf_convolution_cache:
            psf_convolution = self.__psf_convolution_cache[index]

            # Make sure that it is initialized.
            scale_factor = 1.0
            fail_count = 0
            while psf_convolution is None:
                fail_count += 1
                if fail_count > 5:
                    logger.warn("Second instance of %s convolution "
                                "waiting for initialization." %
                                ((str[i] for i in index),))
                time.sleep(0.1*scale_factor)
                scale_factor *= 1.2
                psf_convolution = self.__psf_convolution_cache[index]

            return psf_convolution

        # print "Generating convolution %s" % ((str[i] for i in index),)

        # Mark that we are making the new object
        self.__psf_convolution_cache[index] = None

        # Get the convolution
        psf_convolution = psf1.get_convolution_spline(psf2)

        # Add it to the cache
        self.__psf_convolution_cache[index] = psf_convolution

        return psf_convolution

    def _calc_convolution_matrix(self, psfs_1, psf_counts_1, ras_1, decs_1,
                                 psfs_2, psf_counts_2, ras_2, decs_2):
        """Calculate the GG, GH, HH convolution matrices"""

        diff_ra = np.subtract.outer(ras_1, ras_2)
        diff_dec = np.subtract.outer(decs_1, decs_2)
        matrix = np.zeros((len(ras_1), len(ras_2)))

        offset_1 = 0
        for psf_1, psf_count_1 in zip(psfs_1, psf_counts_1):
            offset_2 = 0
            for psf_2, psf_count_2 in zip(psfs_2, psf_counts_2):
                spline = self._get_psf_convolution(psf_1, psf_2)

                matrix[offset_1:offset_1+psf_count_1,
                       offset_2:offset_2+psf_count_2] = (
                           spline.ev(
                               diff_ra[offset_1:offset_1+psf_count_1,
                                       offset_2:offset_2+psf_count_2],
                               diff_dec[offset_1:offset_1+psf_count_1,
                                        offset_2:offset_2+psf_count_2]
                           )
                )

                offset_2 += psf_count_2

            offset_1 += psf_count_1

        return matrix

    def _do_thread_subtraction(self, sub_grid, sub_grid_shape, err_grid,
                               err_grid_shape, ra_grid, dec_grid, start_index,
                               end_index, radius):
        if isinstance(sub_grid, sharedctypes.SynchronizedArray):
            sub_grid = np.frombuffer(sub_grid.get_obj())
            sub_grid = sub_grid.reshape(sub_grid_shape)
        if isinstance(err_grid, sharedctypes.SynchronizedArray):
            err_grid = np.frombuffer(err_grid.get_obj())
            err_grid = err_grid.reshape(err_grid_shape)

        sub_iter = sub_grid[start_index:end_index, :].flat
        err_iter = err_grid[start_index:end_index, :].flat
        ra_iter = ra_grid[start_index:end_index, :].flat
        dec_iter = dec_grid[start_index:end_index, :].flat

        num_pixels = len(sub_iter)

        print "Start/end: %d %d" % (start_index, end_index)

        for i in range(num_pixels):
            ra = ra_iter[i]
            dec = dec_iter[i]

            # Get all of the pixels within the desired radius
            new_idx = self.new_kdtree.query_ball_point(
                (ra, dec), radius
            )

            new_psfs = self.new_psfs[new_idx]
            new_order = new_psfs.argsort()
            new_psfs = new_psfs[new_order]
            new_vals = self.new_vals[new_idx][new_order]
            new_errs = self.new_errs[new_idx][new_order]
            new_ras = self.new_ras[new_idx][new_order]
            new_decs = self.new_decs[new_idx][new_order]

            ref_idx = self.ref_kdtree.query_ball_point(
                (ra, dec), radius
            )

            ref_psfs = self.ref_psfs[ref_idx]
            ref_order = ref_psfs.argsort()
            ref_vals = self.ref_vals[ref_idx][ref_order]
            ref_errs = self.ref_errs[ref_idx][ref_order]
            ref_ras = self.ref_ras[ref_idx][ref_order]
            ref_decs = self.ref_decs[ref_idx][ref_order]

            # Skip if there wasn't anything
            if len(ref_psfs) == 0 or len(new_psfs) == 0:
                sub_iter[i] = 0.
                err_iter[i] = 0.
                continue

            # Calculate the GG, GH and HH matrices. Note that we have
            # everything grouped by PSF for speed.
            new_psf_objs, new_psf_counts = np.unique(
                new_psfs,
                return_counts=True
            )
            ref_psf_objs, ref_psf_counts = np.unique(
                ref_psfs,
                return_counts=True
            )
            gg = self._calc_convolution_matrix(
                new_psfs, new_psf_counts, new_ras, new_decs,
                new_psfs, new_psf_counts, new_ras, new_decs
            )
            gh = self._calc_convolution_matrix(
                new_psfs, new_psf_counts, new_ras, new_decs,
                ref_psfs, ref_psf_counts, ref_ras, ref_decs
            )
            hh = self._calc_convolution_matrix(
                ref_psfs, ref_psf_counts, ref_ras, ref_decs,
                ref_psfs, ref_psf_counts, ref_ras, ref_decs
            )

            gsn = np.zeros(len(new_psfs))
            offset = 0
            diff_ra = new_ras - ra
            diff_dec = new_decs - dec
            for psf, psf_count in zip(new_psfs, new_psf_counts):
                gsn[offset:offset+psf_count] = psf.psf_spline.ev(
                    diff_ra[offset:offset+psf_count],
                    diff_dec[offset:offset+psf_count]
                )
                offset += psf_count

            hsn = np.zeros(len(ref_psfs))
            offset = 0
            diff_ra = ref_ras - ra
            diff_dec = ref_decs - dec
            for psf, psf_count in zip(ref_psfs, ref_psf_counts):
                hsn[offset:offset+psf_count] = psf.psf_spline.ev(
                    diff_ra[offset:offset+psf_count],
                    diff_dec[offset:offset+psf_count]
                )
                offset += psf_count

            new_n = np.diag(new_errs**2)
            ref_n = np.diag(ref_errs**2)

            ref_val = hsn.dot(ref_vals) / np.sum(hsn)
            ref_err = np.median(ref_errs)
            new_err = np.median(new_errs)
            #f = np.median(ref_vals) + np.median(ref_errs)
            #amp = 5.*np.median(ref_errs) / gsn.dot(gsn)
            if ref_val < 0:
                ref_val = 0
            f = ref_val + ref_err
            amp = 5*f + 20.*new_err
            ref_amp = amp

            a = gg*f**2 + amp**2 * np.outer(gsn, gsn) + new_n
            b = -2. * amp**2 * gsn
            c = -2. * gh*f**2
            d = hh*f**2 + ref_amp**2 * np.outer(hsn, hsn) + ref_n
            e = -2. * ref_amp**2 * hsn

            d_inv = np.linalg.inv(d)

            #t = - np.linalg.inv(2*a - 1/2.*c.dot(d_inv.dot(c.T))).dot(b)
            #s = -1/2. * d_inv.dot(c.T.dot(t).T)
            t = (
                (1/2. * e.dot(d_inv).dot(c.T) - b)
                .dot(np.linalg.inv(2.*a - 1/2.*c.dot(d_inv.dot(c.T))))
            )
            s = -1/2. * (t.dot(c) + e).dot(d_inv)

            sub = t.dot(new_vals) - s.dot(ref_vals)
            err = np.sqrt(
                (a.dot(t).dot(t)
                 + b.dot(t)
                 + c.dot(s).dot(t)
                 + d.dot(s).dot(s)
                 + e.dot(s)
                 + amp*amp
                 + ref_amp*ref_amp)
            )

            #from IPython import embed; embed()

            sub_iter[i] = sub
            err_iter[i] = err

            #center_ra = (self.output_center_ra
                         #* np.cos(self.output_center_dec * np.pi / 180.)
                         #* 3600.)
            #center_dec = self.output_center_dec * 3600.
            #psf = new_psfs[0]
            #sub_iter[i] += 1000*psf.psf_spline.ev(ra - center_ra, dec
                                                 #- center_dec)

        #print sub_grid
