from astropy.io import fits
import sys
import numpy as np
from seechange.data import FitsFile

base = '/home/scpdata05/clustersn/kboone/idcsj/data/'

scale_image = FitsFile.open(
    base + 'SPT0205/sets/SPT0205_v1_F140W.fits',
    readonly=True
)

#scale_image = FitsFile.open(
    #base + 'SPARCS-J003550-431210/sets/SPARCS-J003550-431210_v1_F140W.fits',
    #readonly=True
#)

naxis1 = scale_image.naxis1[0]
naxis2 = scale_image.naxis2[0]

if __name__ == "__main__":
    args = sys.argv

    try:
        num_tiles_x = int(args[1])
        num_tiles_y = int(args[2])
    except:
        print "Invalid arguments, format is:"
        print "%s [num_tiles_x] [num_tiles_y]" % args[0]
        sys.exit(1)

    top_hdu = scale_image.hdulist[0]
    hdus = [top_hdu]
    all_datas = []

    for image_type in ['sub', 'err']:
        col_data = []
        for i in range(num_tiles_x):
            row_data = []
            for j in range(num_tiles_x):
                data = np.load('./grid_%s_%d_%d.npy' % (image_type, i, j))
                row_data.append(data)

            col_data.append(np.vstack(row_data))

        all_data = np.hstack(col_data)
        all_datas.append(all_data)

        new_hdu = fits.ImageHDU(
            data=all_data,
            header=scale_image.hdulist[1].header
        )

        hdus.append(new_hdu)

    # significance
    data = all_datas[0]
    err = all_datas[1]
    mask = err == 0
    err[mask] = 1.
    sig_data = data / np.sqrt(err)
    sig_data[all_data[1] == 0] = 0.
    sig_hdu = fits.ImageHDU(
        data=sig_data,
        header=scale_image.hdulist[1].header
    )
    hdus.append(sig_hdu)

    hdulist = fits.HDUList(hdus)
    hdulist.writeto('./output_bob2.fits')
