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
from seechange.data import FitsFile

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
        fits_file = FitsFile.open(path, readonly=True)

        hdulist = fits_file.hdulist

        sci_header = hdulist[extension].header

        self.output_naxis1 = sci_header['NAXIS1']
        self.output_naxis2 = sci_header['NAXIS2']

        wcs = WCS(sci_header, fobj=hdulist)
        self.output_center_ra, self.output_center_dec = wcs.wcs.crval

        cd_matrix = wcs.pixel_scale_matrix
        pixel_scale = 3600.*np.sqrt(cd_matrix[0, 0]**2 + cd_matrix[0, 1]**2)
        self.output_pixel_scale = pixel_scale

        # Test: object detection
        import sep
        data = fits_file.get_data(copy=True)
        mask = fits_file.get_mask()
        sep_sky = fits_file.calc_sep_sky()
        flat_sky_rms = sep_sky.globalrms
        threshold = 5. * flat_sky_rms

        # TODO: update to new sep which can use a variable mask
        data[mask] = 0.
        objects = sep.extract(data, threshold)
        objects = objects[objects['npix'] > 10]
        self.bright_objects = objects

        obj_im_xs = objects['x'] + 1
        obj_im_ys = objects['y'] + 1
        #obj_im_xs[0] = 50.
        #obj_im_ys[0] = 75.
        obj_ras, obj_decs = fits_file.xy_to_rd(obj_im_xs, obj_im_ys)

        self.bright_obj_ras = obj_ras
        self.bright_obj_decs = obj_decs

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
        # Note that we convert all of our ra and dec values into x and y
        # coordinates in arcseconds relative to (0., 0.). We flip the x-axis so
        # that our coordinates increase from left to right (or East to West).
        # TODO: We use double the memory that we need to currently. This should
        # be optimized to query all of the files for how many points they have,
        # and then load them one by one into preallocated arrays.
        self.ref_vals = np.hstack([i.vals for i in self.ref_images])
        self.ref_errs = np.hstack([i.errs for i in self.ref_images])
        self.ref_psfs = np.hstack([i.psfs for i in self.ref_images])
        self.ref_xs = np.hstack([i.ras for i in self.ref_images])
        self.ref_xs *= -np.cos(self.output_center_dec * np.pi / 180.) * 3600.
        self.ref_ys = np.hstack([i.decs for i in self.ref_images])
        self.ref_ys *= 3600.

        self.new_vals = np.hstack([i.vals for i in self.new_images])
        self.new_errs = np.hstack([i.errs for i in self.new_images])
        self.new_psfs = np.hstack([i.psfs for i in self.new_images])
        self.new_xs = np.hstack([i.ras for i in self.new_images])
        self.new_xs *= -np.cos(self.output_center_dec * np.pi / 180.) * 3600.
        self.new_ys = np.hstack([i.decs for i in self.new_images])
        self.new_ys *= 3600.

        # Generate KDTrees.
        # TODO: Try to use a single KDTree for both? Profile this, and if the
        # KDTree is limiting then that might be a good idea.
        print "Generating KDTrees"
        self.ref_kdtree = cKDTree(zip(self.ref_xs, self.ref_ys))
        self.new_kdtree = cKDTree(zip(self.new_xs, self.new_ys))

        # Load the bright objects
        self.bright_obj_xs = (-np.cos(self.output_center_dec * np.pi / 180.) *
                              3600. * self.bright_obj_ras)
        self.bright_obj_ys = self.bright_obj_decs * 3600.
        bright_obj_fluxes = [self._calc_value(x, y, radius=0.3, is_star=True)
                             for x, y in zip(self.bright_obj_xs,
                                             self.bright_obj_ys)]
        self.bright_obj_fluxes = np.array(bright_obj_fluxes)

        print "Done loading data"

    def do_subtraction(self, num_cores=8, num_tiles_x=1, num_tiles_y=1,
                       x_tile=0, y_tile=0, radius=0.3):
        """Do the subtraction.

        This can be done with tiles to easily split the work between different
        machines. The default settings will do the full image.
        """

        # Figure out the pixel indices that we need to run on. Note that the
        # indices are 1 based.
        num_per_tile_x = int(np.ceil(self.output_naxis1 / float(num_tiles_x)))
        start_pix_x = x_tile * num_per_tile_x + 1
        end_pix_x = (x_tile + 1) * num_per_tile_x + 1
        if end_pix_x > self.output_naxis1:
            end_pix_x = self.output_naxis1 + 1

        num_per_tile_y = int(np.ceil(self.output_naxis2 / float(num_tiles_y)))
        start_pix_y = y_tile * num_per_tile_y + 1
        end_pix_y = (y_tile + 1) * num_per_tile_y + 1
        if end_pix_y > self.output_naxis2:
            end_pix_y = self.output_naxis2 + 1

        x_pix_range = np.arange(start_pix_x, end_pix_x)
        y_pix_range = np.arange(start_pix_y, end_pix_y)

        # Get the coordinates on our x/y arcsecond grid.
        x_pix_grid, y_pix_grid = np.meshgrid(x_pix_range, y_pix_range)
        x_grid, y_grid = self.output_wcs.all_pix2world(
            x_pix_grid,
            y_pix_grid,
            1
        )
        x_grid *= -np.cos(self.output_center_dec * np.pi / 180.) * 3600.
        y_grid *= 3600.

        # Split up the work on all of the cores that we have available
        # Note: using threads kind of sucks because the spline interpolation
        # which is our time limiting factor doesn't disable the GIL. Only
        # processes actually work for speedup now.
        use_threads = False
        if num_cores > 1:
            if use_threads:
                # Use multithreading
                sub_grid = np.zeros(x_grid.shape)
                err_grid = np.zeros(x_grid.shape)

                threads = []
                for i in range(num_cores):
                    num_per_core = int(np.ceil(len(y_pix_range) /
                                               float(num_cores)))
                    core_start_index = i * num_per_core
                    core_end_index = (i + 1) * num_per_core
                    if core_end_index > len(y_pix_range):
                        core_end_index = len(y_pix_range)

                    thread = threading.Thread(
                        target=self._do_thread_subtraction,
                        args=(sub_grid, sub_grid.shape, err_grid,
                              err_grid.shape, x_grid, y_grid,
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

                sub_grid_shared = mp.Array('d', np.product(x_grid.shape))
                err_grid_shared = mp.Array('d', np.product(x_grid.shape))

                sub_grid = np.frombuffer(sub_grid_shared.get_obj())
                sub_grid = sub_grid.reshape(x_grid.shape)
                err_grid = np.frombuffer(err_grid_shared.get_obj())
                err_grid = err_grid.reshape(x_grid.shape)

                for i in range(num_cores):
                    num_per_core = int(np.ceil(len(y_pix_range) /
                                               float(num_cores)))
                    core_start_index = i * num_per_core
                    core_end_index = (i + 1) * num_per_core
                    if core_end_index > len(y_pix_range):
                        core_end_index = len(y_pix_range)

                    process = mp.Process(
                        target=self.__class__._do_thread_subtraction,
                        args=(self, sub_grid_shared, sub_grid.shape,
                              err_grid_shared, err_grid.shape, x_grid,
                              y_grid, core_start_index, core_end_index,
                              radius)
                    )
                    processes.append(process)
                    process.start()

                # Wait for our threads to end
                for process in processes:
                    process.join()
        else:
            # Single core, just call it directly
            sub_grid = np.zeros(x_grid.shape)
            err_grid = np.zeros(y_grid.shape)

            self._do_thread_subtraction(
                sub_grid, sub_grid.shape, err_grid, err_grid.shape, x_grid,
                y_grid, 0, len(y_pix_range), radius
            )

        np.save('./grid_sub_%d_%d.npy' % (x_tile, y_tile), sub_grid)
        np.save('./grid_err_%d_%d.npy' % (x_tile, y_tile), err_grid)

        return sub_grid, err_grid

    __psf_convolution_cache = {}
    __psf_taylor_cache = {}

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

    def _get_psf_taylor_spline(self, psf, order_x, order_y):
        """Get the convolution of a PSF with (x^order_x*y^order_y)

        We use a cache so that calling this multiple times actually returns the
        same object.
        """
        # We optimize and return copies of the same object. This is done by
        # maintaining a cache of opened objects, and checking if it is in
        # there.
        index = (psf, order_x, order_y)
        if index in self.__psf_taylor_cache:
            spline = self.__psf_taylor_cache[index]

            # Make sure that it is initialized.
            scale_factor = 1.0
            fail_count = 0
            while spline is None:
                fail_count += 1
                if fail_count > 5:
                    logger.warn("Second instance of %s convolution "
                                ((str[i] for i in index),))
                time.sleep(0.1*scale_factor)
                scale_factor *= 1.2
                spline = self.__psf_taylor_cache[index]

            return spline

        # print "Generating convolution %s" % ((str[i] for i in index),)

        # Mark that we are making the new object
        self.__psf_taylor_cache[index] = None

        # Get the convolution
        spline = psf.get_taylor_spline(order_x, order_y)

        # Add it to the cache
        self.__psf_taylor_cache[index] = spline

        return spline

    def _calc_psf_taylor_vector(self, psfs, psf_counts, xs, ys, center_x,
                                center_y, order_x, order_y):
        """Calculate the PSF Taylor vector of a given order"""
        result = np.zeros(len(psfs))
        diff_x = xs - center_x
        diff_y = ys - center_y
        offset = 0
        for psf, psf_count in zip(psfs, psf_counts):
            spline = self._get_psf_taylor_spline(psf, order_x, order_y)
            result[offset:offset+psf_count] = spline.ev(
                diff_y[offset:offset+psf_count],
                diff_x[offset:offset+psf_count]
            )
            offset += psf_count

        return result

    def _calc_convolution_matrix(self, psfs_1, psf_counts_1, xs_1, ys_1,
                                 psfs_2, psf_counts_2, xs_2, ys_2):
        """Calculate the GG, GH, HH convolution matrices"""

        diff_x = np.subtract.outer(xs_1, xs_2)
        diff_y = np.subtract.outer(ys_1, ys_2)
        matrix = np.zeros((len(xs_1), len(xs_2)))

        offset_1 = 0
        for psf_1, psf_count_1 in zip(psfs_1, psf_counts_1):
            offset_2 = 0
            for psf_2, psf_count_2 in zip(psfs_2, psf_counts_2):
                spline = self._get_psf_convolution(psf_1, psf_2)

                matrix[offset_1:offset_1+psf_count_1,
                       offset_2:offset_2+psf_count_2] = (
                           spline.ev(
                               diff_y[offset_1:offset_1+psf_count_1,
                                      offset_2:offset_2+psf_count_2],
                               diff_x[offset_1:offset_1+psf_count_1,
                                      offset_2:offset_2+psf_count_2]
                           )
                )

                offset_2 += psf_count_2

            offset_1 += psf_count_1

        return matrix

    def _calc_cross_matrix(self, x, y, psfs_1, psf_counts_1, xs_1, ys_1,
                           psfs_2, psf_counts_2, xs_2, ys_2, offset=0.2):
        """Calculate the GG, GH, HH matrices"""

        # Estimate derivatives. We first calculate points on the following
        # grid, and use the finite difference method
        # a b c
        # d e f
        # g h i
        f_a = self._calc_value(x-offset, y+offset)
        f_b = self._calc_value(x,        y+offset)
        f_c = self._calc_value(x+offset, y+offset)
        f_d = self._calc_value(x-offset, y)
        f_e = self._calc_value(x,        y)
        f_f = self._calc_value(x+offset, y)
        f_g = self._calc_value(x-offset, y-offset)
        f_h = self._calc_value(x,        y-offset)
        f_i = self._calc_value(x+offset, y-offset)

        f = f_e
        f_x = (f_f - f_d) / (2 * offset)
        f_y = (f_b - f_h) / (2 * offset)
        f_xx = (f_f - 2*f_e + f_d) / (offset * offset)
        f_yy = (f_b - 2*f_e + f_h) / (offset * offset)
        f_xy = (f_c + f_g - f_a - f_i) / (4 * offset * offset)

        coeffs_1 = np.zeros(len(psfs_1))
        coeffs_2 = np.zeros(len(psfs_2))

        # print x, y, f, f_x, f_y, f_xx, f_yy, f_xy

        terms = [
            (0, 0, f),
            (1, 0, f_x),
            (0, 1, f_y),
            (2, 0, f_xx / 2.),
            (1, 1, f_xy / 2.),
            (0, 2, f_yy),
        ]

        for order_x, order_y, factor in terms:
            vec_1 = self._calc_psf_taylor_vector(
                psfs_1, psf_counts_1, xs_1, ys_1, x, y, order_x, order_y
            )
            vec_2 = self._calc_psf_taylor_vector(
                psfs_2, psf_counts_2, xs_2, ys_2, x, y, order_x, order_y
            )
            # print order_x, order_y, factor, np.min(vec_1), np.max(vec_1)
            coeffs_1 += factor * vec_1
            coeffs_2 += factor * vec_2

        return np.outer(coeffs_1, coeffs_2)

    def _calc_psf_vector(self, psfs, psf_counts, xs, ys, center_x, center_y):
        """Calculate a PSF vector (gsn, hsn, etc.)"""
        psf_vector = np.zeros(len(xs))
        diff_x = xs - center_x
        diff_y = ys - center_y
        offset = 0
        for psf, psf_count in zip(psfs, psf_counts):
            psf_vector[offset:offset+psf_count] = psf.psf_spline.ev(
                diff_y[offset:offset+psf_count],
                diff_x[offset:offset+psf_count]
            )
            offset += psf_count

        return psf_vector

    def _calc_value(self, x, y, radius=0.1, is_star=False):
        """Calculate an approximate value of the underlying data function

        If is_star is True, then the object is treated as a star (i.e. a single
        point) instead of a smooth underlying function.

        This is smeared out by the PSF, so it isn't really right...
        """
        idx = self.ref_kdtree.query_ball_point((x, y), radius)
        psfs = self.ref_psfs[idx]
        order = psfs.argsort()
        psfs = psfs[order]
        vals = self.ref_vals[idx][order]
        xs = self.ref_xs[idx][order]
        ys = self.ref_ys[idx][order]

        psf_objs, psf_counts = np.unique(
            psfs,
            return_counts=True
        )

        weights = self._calc_psf_vector(
            psf_objs, psf_counts, xs, ys, x, y
        )

        if is_star:
            val = weights.dot(vals) / weights.dot(weights)
        else:
            val = weights.dot(vals) / np.sum(weights)

        return val

    def weighted_median(values, weights):
        if len(values) == 1:
            return values[0]
        if len(values) == 0:
            raise Exception("Can't do weighted median of nothing!")
        order = np.argsort(values)
        values = values[order]
        weights = weights[order]
        abs_weights = np.abs(weights)
        cumsum = np.cumsum(abs_weights)
        mid = cumsum[-1] / 2.
        centers = cumsum - abs_weights / 2.
        index = np.searchsorted(centers, mid, side='right')

        val_left = values[index-1]
        val_right = values[index]
        center_left = centers[index-1]
        center_right = centers[index]
        center_diff = center_right - center_left

        weight_right = (mid - center_left) / center_diff
        weight_left = (center_right - mid) / center_diff

        result = weight_left * val_left + weight_right * val_right
        result *= np.sum(weights)

        return result

    def _do_thread_subtraction(self, sub_grid, sub_grid_shape, err_grid,
                               err_grid_shape, x_grid, y_grid, start_index,
                               end_index, radius):
        if isinstance(sub_grid, sharedctypes.SynchronizedArray):
            sub_grid = np.frombuffer(sub_grid.get_obj())
            sub_grid = sub_grid.reshape(sub_grid_shape)
        if isinstance(err_grid, sharedctypes.SynchronizedArray):
            err_grid = np.frombuffer(err_grid.get_obj())
            err_grid = err_grid.reshape(err_grid_shape)

        sub_iter = sub_grid[start_index:end_index, :].flat
        err_iter = err_grid[start_index:end_index, :].flat
        x_iter = x_grid[start_index:end_index, :].flat
        y_iter = y_grid[start_index:end_index, :].flat

        num_pixels = len(sub_iter)

        print "Start/end: %d %d" % (start_index, end_index)

        for i in range(num_pixels):
            x = x_iter[i]
            y = y_iter[i]

            # Get all of the pixels within the desired radius
            new_idx = self.new_kdtree.query_ball_point(
                (x, y), radius
            )

            new_psfs = self.new_psfs[new_idx]
            new_order = new_psfs.argsort()
            new_psfs = new_psfs[new_order]
            new_vals = self.new_vals[new_idx][new_order]
            new_errs = self.new_errs[new_idx][new_order]
            new_xs = self.new_xs[new_idx][new_order]
            new_ys = self.new_ys[new_idx][new_order]

            ref_idx = self.ref_kdtree.query_ball_point(
                (x, y), radius
            )

            ref_psfs = self.ref_psfs[ref_idx]
            ref_order = ref_psfs.argsort()
            ref_psfs = ref_psfs[ref_order]
            ref_vals = self.ref_vals[ref_idx][ref_order]
            ref_errs = self.ref_errs[ref_idx][ref_order]
            ref_xs = self.ref_xs[ref_idx][ref_order]
            ref_ys = self.ref_ys[ref_idx][ref_order]

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
                new_psf_objs, new_psf_counts, new_xs, new_ys,
                new_psf_objs, new_psf_counts, new_xs, new_ys
            )
            gh = self._calc_convolution_matrix(
                new_psf_objs, new_psf_counts, new_xs, new_ys,
                ref_psf_objs, ref_psf_counts, ref_xs, ref_ys
            )
            hh = self._calc_convolution_matrix(
                ref_psf_objs, ref_psf_counts, ref_xs, ref_ys,
                ref_psf_objs, ref_psf_counts, ref_xs, ref_ys
            )

            gsn = self._calc_psf_vector(
                new_psf_objs, new_psf_counts, new_xs, new_ys, x, y
            )
            hsn = self._calc_psf_vector(
                ref_psf_objs, ref_psf_counts, ref_xs, ref_ys, x, y
            )

            new_n = np.diag(new_errs**2)
            ref_n = np.diag(ref_errs**2)

            ref_val = hsn.dot(ref_vals) / np.sum(hsn)
            ref_err = np.median(ref_errs)
            new_err = np.median(new_errs)
            #f = np.median(ref_vals) + np.median(ref_errs)
            #amp = 5.*np.median(ref_errs) / gsn.dot(gsn)
            if ref_val < 0:
                ref_val = 0
            #f = ref_val / 10.
            #f = self._calc_value(x, y, radius=0.3, is_star=True)
            f = 0.
            #f = 1.
            amp = 100.0#ref_val + ref_err + new_err
            #ref_amp = amp
            ref_amp = 100.

            #ref_n += np.diag(np.ones(len(ref_n)) * (ref_val / 100.)**2)
            #new_n += np.diag(np.ones(len(new_n)) * (ref_val / 100.)**2)

            a = gg*f**2 + amp**2 * np.outer(gsn, gsn) + new_n
            b = -2. * amp**2 * gsn
            c = -2. * gh*f**2
            d = hh*f**2 + ref_amp**2 * np.outer(hsn, hsn) + ref_n
            e = -2. * ref_amp**2 * hsn

            # Bright objects
            for obj_x, obj_y, obj_flux in zip(self.bright_obj_xs,
                                              self.bright_obj_ys,
                                              self.bright_obj_fluxes):
                #if (x - obj_x)**2 + (y - obj_y)**2 > 4.:
                    #continue
                g_bright_obj = self._calc_psf_vector(
                    new_psf_objs, new_psf_counts, new_xs, new_ys, obj_x, obj_y
                )
                h_bright_obj = self._calc_psf_vector(
                    ref_psf_objs, ref_psf_counts, ref_xs, ref_ys, obj_x, obj_y
                )

                a += obj_flux*obj_flux*np.outer(g_bright_obj, g_bright_obj)
                c += -2. * obj_flux*obj_flux*np.outer(g_bright_obj, h_bright_obj)
                d += obj_flux*obj_flux*np.outer(h_bright_obj, h_bright_obj)

                #a += np.diag((0.01*obj_flux*g_bright_obj)**2)
                #d += np.diag((0.01*obj_flux*h_bright_obj)**2)

            d_inv = np.linalg.inv(d)

            #t = - np.linalg.inv(2*a - 1/2.*c.dot(d_inv.dot(c.T))).dot(b)
            #s = -1/2. * d_inv.dot(c.T.dot(t).T)
            t = (
                (1/2. * e.dot(d_inv).dot(c.T) - b)
                .dot(np.linalg.inv(2.*a - 1/2.*c.dot(d_inv.dot(c.T))))
            )
            s = -1/2. * (t.dot(c) + e).dot(d_inv)

            sub = t.dot(new_vals) - s.dot(ref_vals)

            #print new_sub_val, t.dot(new_vals), ref_sub_val, s.dot(ref_vals), \
                #new_sub_val - ref_sub_val, new_sub, old_sub

            # Remove amps from the final err, they are just there for
            # normalization
            a = gg*f**2 + new_n
            c = -2. * gh*f**2
            d = hh*f**2 + ref_n

            # Bright objects
            for obj_x, obj_y, obj_flux in zip(self.bright_obj_xs,
                                              self.bright_obj_ys,
                                              self.bright_obj_fluxes):
                #if (x - obj_x)**2 + (y - obj_y)**2 > 4.:
                    #continue
                g_bright_obj = self._calc_psf_vector(
                    new_psf_objs, new_psf_counts, new_xs, new_ys, obj_x, obj_y
                )
                h_bright_obj = self._calc_psf_vector(
                    ref_psf_objs, ref_psf_counts, ref_xs, ref_ys, obj_x, obj_y
                )

                a += obj_flux*obj_flux*np.outer(g_bright_obj, g_bright_obj)
                c += -2. * obj_flux*obj_flux*np.outer(g_bright_obj, h_bright_obj)
                d += obj_flux*obj_flux*np.outer(h_bright_obj, h_bright_obj)

                #a += np.diag((0.01*obj_flux*g_bright_obj)**2)
                #d += np.diag((0.01*obj_flux*h_bright_obj)**2)


            err = np.sqrt(
                (a.dot(t).dot(t)
                 + c.dot(s).dot(t)
                 + d.dot(s).dot(s)
                 )
            )

            #err = np.sqrt(
                #(a.dot(t).dot(t)
                 #+ b.dot(t)
                 #+ c.dot(s).dot(t)
                 #+ d.dot(s).dot(s)
                 #+ e.dot(s)
                 #+ ref_amp*ref_amp
                 #+ amp*amp
                 #)
            #)

            debug = 0

            if debug:
                var_f = (
                    (gg*f**2).dot(t).dot(t)
                    - 2*(gh*f**2).dot(s).dot(t)
                    + (hh*f**2).dot(s).dot(s)
                )
                var_gsn = (
                    amp**2*np.outer(gsn, gsn).dot(t).dot(t)
                    -2.*amp**2*gsn.dot(t)
                    + amp*amp
                )
                var_hsn = (
                    ref_amp**2*np.outer(hsn, hsn).dot(s).dot(s)
                    -2.*ref_amp**2*hsn.dot(s)
                    + ref_amp*ref_amp
                )
                var_refn = (
                    ref_n.dot(s).dot(s)
                )
                var_newn = (
                    new_n.dot(t).dot(t)
                )

                print "total var: %f -> check %f" % (err*err, var_f + var_gsn +
                                                     var_hsn + var_refn +
                                                     var_newn)
                print "    var_f:   %f" % var_f
                print "    var_gsn: %f" % var_gsn
                print "    var_hsn: %f" % var_hsn
                print "    var_refn: %f" % var_refn
                print "    var_newn: %f" % var_newn
                print "    ---"
                print "    sub: %f" % sub
                print "    err: %f" % err

                from IPython import embed; embed()

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
