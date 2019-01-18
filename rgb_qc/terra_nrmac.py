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
from pyclowder.datasets import download_metadata as download_ds_metadata
from terrautils.extractors import TerrarefExtractor, build_metadata, upload_to_dataset
from terrautils.metadata import get_extractor_metadata, get_terraref_metadata
from terrautils.formats import create_geotiff
from terrautils.spatial import geojson_to_tuples


def MAC(self, im): # main function: Multiscale Autocorrelation (MAC)
    h, v, c = im.shape
    if c>1:
        im  = np.matrix.round(rgb2gray(im))
    # multiscale parameters
    scales = np.array([2, 3, 5])
    dif = np.zeros(len(scales))
    for s in range(len(scales)):
        # part 1 image
        f11 = im[0:h-1,:]
        f12 = im[1:,:]
        # part 2 image
        f21 = im[0:h-scales[s],:]
        f22 = im[scales[s]:,:]
        f1 = f11*f12
        f2 = f21*f22
        # sum and compute difference
        dif[s] = np.sum(f1) - np.sum(f2)
    NRMAC = np.mean(dif)
    return NRMAC

def rgb2gray(self, rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def getImageQuality(self, imgfile):
    img = Image.open(imgfile)
    img = np.array(img)
    NRMAC = MAC(img)
    return NRMAC


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
        qual = getImageQuality(f)

        self.log_info(resource, "creating output image")
        md = download_ds_metadata(connector,host, secret_key, resource['parent']['id'])
        terramd = get_terraref_metadata(md)
        if "left" in f:
            bounds = geojson_to_tuples(terramd['spatial_metadata']['left']['bounding_box'])
        else:
            bounds = geojson_to_tuples(terramd['spatial_metadata']['right']['bounding_box'])
        output = f.replace(".tif", "_nrmac.tif")
        create_geotiff(np.array([[qual,qual],[qual,qual]]), bounds, output)
        upload_to_dataset(connector, host, self.clowder_user, self.clowder_pass, resource['parent']['id'], output)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        ext_meta = build_metadata(host, self.extractor_info, resource['id'], {
            "quality_score": qual
        }, 'file')
        self.log_info(resource, "uploading extractor metadata")
        upload_metadata(connector, host, secret_key, resource['id'], ext_meta)

        self.end_message(resource)



if __name__ == "__main__":
    extractor = RGB_NRMAC()
    extractor.start()
