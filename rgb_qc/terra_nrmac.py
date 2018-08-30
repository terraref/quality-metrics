"""
# Name:        No-Reference Multiscale Autocorrelation (NRMAC)
# Purpose:     To quantify the image quality based on a No-Reference Multiscale Autocorrelation Metric.
#              References: This method is a modification of the Vollath's correlation (Santos, 1997) metric
#              Function: This NRMAC is a focus measure based on image autocorrelation from multiple scales
#              The output with a lower value indicates poorer quality of the input image. 
               The NRMAC has been tested on RGB geotif images and an empirical threshold is set be 15 and needs to be further evaluated. This value is subject to be changed based on sensors setting and user requirements. 
# Version:     1.0
#
# Created:     03/20/2018
"""

import numpy as np
from PIL import Image, ImageFilter

from pyclowder.utils import CheckMessage
from pyclowder.files import download_metadata, upload_metadata
from terrautils.extractors import TerrarefExtractor, build_metadata
from terrautils.metadata import get_extractor_metadata


class RGB_NRMAC(TerrarefExtractor):
    def __init__(self):
        super(RGB_NRMAC, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor='rgb_geotiff')

    def check_message(self, connector, host, secret_key, resource, parameters):
        if "rulechecked" in parameters and parameters["rulechecked"]:
            return CheckMessage.download
        self.start_check(resource)

        if resource['name'].endswith('_left.tif') or resource['name'].endswith('_right.tif'):
            # Check metadata to verify we have what we need
            md = download_metadata(connector, host, secret_key, resource['id'])
            if get_extractor_metadata(md, self.extractor_info['name']) and not self.overwrite:
                self.log_skip(resource, "metadata indicates it was already processed")
                return CheckMessage.ignore
            return CheckMessage.download
        else:
            self.log_skip(resource, "not left/right geotiff")
            return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message(resource)

        f = resource['local_paths'][0]

        self.log_info(resource, "determining image quality")
        qual = self.getImageQuality(f)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        ext_meta = build_metadata(host, self.extractor_info, resource['id'], {
            "quality_score": qual
        }, 'file')
        self.log_info(resource, "uploading extractor metadata")
        upload_metadata(connector, host, secret_key, resource['id'], ext_meta)

        self.end_message(resource)

    def MAC(self, im1,im2, im): # main function: Multiscale Autocorrelation (MAC)
        h, v, c = im1.shape
        if c>1:
            im  = np.matrix.round(self.rgb2gray(im))
            im1 = np.matrix.round(self.rgb2gray(im1))
            im2 = np.matrix.round(self.rgb2gray(im2))
            # multiscale parameters
        scales = np.array([2, 3, 5])
        FM = np.zeros(len(scales))
        for s in range(len(scales)):
            im1[0: h-1,:] = im[1:h,:]
            im2[0: h-scales[s], :]= im[scales[s]:h,:]
            dif = im*(im1 - im2)
            FM[s] = np.mean(dif)
        NRMAC = np.mean(FM)
        return NRMAC

    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def getImageQuality(self, imgfile):
        img = Image.open(imgfile)
        img = np.array(img)

        NRMAC = self.MAC(img, img, img)

        return NRMAC

if __name__ == "__main__":
    extractor = RGB_NRMAC()
    extractor.start()
