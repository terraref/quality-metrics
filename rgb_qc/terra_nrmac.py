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
import os
import shutil
import tempfile
import cv2
import numpy as np
from osgeo import gdal
from PIL import Image, ImageFilter

from pyclowder.utils import CheckMessage
from pyclowder.datasets import download_metadata, upload_metadata, remove_metadata, submit_extraction
from terrautils.extractors import TerrarefExtractor, build_metadata, upload_to_dataset, \
    is_latest_file, contains_required_files, file_exists, load_json_file, check_file_in_dataset
from terrautils.metadata import get_extractor_metadata, get_terraref_metadata
from terrautils.formats import create_geotiff
from terrautils.spatial import geojson_to_tuples


def MAC(im1, im2, im): # main function: Multiscale Autocorrelation (MAC)
    h, v, c = im1.shape
    if c>1:
        im  = np.matrix.round(rgb2gray(im))
        im1 = np.matrix.round(rgb2gray(im1))
        im2 = np.matrix.round(rgb2gray(im2))
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

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def getImageQuality(imgfile):
    # Some RGB Geotiffs have issues with Image library...
    #img = Image.open(imgfile)
    #img = np.array(img)
    img = np.rollaxis(gdal.Open(imgfile).ReadAsArray().astype(np.uint8), 0, 3)
    NRMAC = MAC(img, img, img)
    return NRMAC


def add_local_arguments(parser):
    # add any additional arguments to parser
    parser.add_argument('--left', type=bool, default=os.getenv('LEFT_ONLY', True),
                        help="only generate a mask for the left image")

class RGB_NRMAC(TerrarefExtractor):
    def __init__(self):
        super(RGB_NRMAC, self).__init__()

        add_local_arguments(self.parser)

        # parse command line and load default logging configuration
        self.setup(sensor='rgb_nrmac')

        # assign local arguments
        self.leftonly = self.args.left

    def check_message(self, connector, host, secret_key, resource, parameters):
        if "rulechecked" in parameters and parameters["rulechecked"]:
            return CheckMessage.download

        self.start_check(resource)

        if not is_latest_file(resource):
            self.log_skip(resource, "not latest file")
            return CheckMessage.ignore

        # Check for a left and right BIN file - skip if not found
        if not contains_required_files(resource, ['_left.tif', '_right.tif']):
            self.log_skip(resource, "missing required files")
            # Check for raw_data_source in metadata and resumbit to bin2tif if available...
            md = download_metadata(connector, host, secret_key, resource['id'])
            terra_md = get_terraref_metadata(md)
            if 'raw_data_source' in terra_md:
                raw_id = str(terra_md['raw_data_source'].split("/")[-1])
                self.log_info(resource, "submitting raw source %s to bin2tif" % raw_id)
                submit_extraction(connector, host, secret_key, raw_id, "terra.stereo-rgb.bin2tif")
            return CheckMessage.ignore

        # Check metadata to verify we have what we need
        md = download_metadata(connector, host, secret_key, resource['id'])
        if get_terraref_metadata(md):
            if get_extractor_metadata(md, self.extractor_info['name'], self.extractor_info['version']):
                # Make sure outputs properly exist
                timestamp = resource['dataset_info']['name'].split(" - ")[1]
                left_nrmac_tiff = self.sensors.create_sensor_path(timestamp, opts=['left'])
                right_nrmac_tiff = self.sensors.create_sensor_path(timestamp, opts=['right'])
                if (self.leftonly and file_exists(left_nrmac_tiff)) or (
                                not self.leftonly and file_exists(left_nrmac_tiff) and file_exists(right_nrmac_tiff)):
                    if contains_required_files(resource, [os.path.basename(left_nrmac_tiff)]):
                        self.log_skip(resource, "metadata v%s and outputs already exist" % self.extractor_info['version'])
                        return CheckMessage.ignore
                    else:
                        self.log_info(resource, "output file exists but not yet uploaded")
            # Have TERRA-REF metadata, but not any from this extractor
            return CheckMessage.download
        else:
            self.log_skip(resource, "no terraref metadata found")
            return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message(resource)

        # Get left/right files and metadata
        img_left, img_right, metadata = None, None, None
        for fname in resource['local_paths']:
            if fname.endswith('_dataset_metadata.json'):
                all_dsmd = load_json_file(fname)
                terra_md_full = get_terraref_metadata(all_dsmd, 'stereoTop')
            elif fname.endswith('_left.tif'):
                img_left = fname
            elif fname.endswith('_right.tif'):
                img_right = fname
        if None in [img_left, img_right, terra_md_full]:
            raise ValueError("could not locate all files & metadata in processing")

        timestamp = resource['dataset_info']['name'].split(" - ")[1]
        target_dsid = resource['id']
        left_nrmac_tiff = self.sensors.create_sensor_path(timestamp, opts=['left'])
        right_nrmac_tiff = self.sensors.create_sensor_path(timestamp, opts=['right'])
        uploaded_file_ids = []

        self.log_info(resource, "determining image quality")
        left_qual = getImageQuality(img_left)
        if not self.leftonly:
            right_qual = getImageQuality(img_right)

        left_bounds = geojson_to_tuples(terra_md_full['spatial_metadata']['left']['bounding_box'])
        right_bounds = geojson_to_tuples(terra_md_full['spatial_metadata']['right']['bounding_box'])

        if not file_exists(left_nrmac_tiff) or self.overwrite:
            self.log_info(resource, "creating %s" % left_nrmac_tiff)
            create_geotiff(np.array([[left_qual, left_qual],[left_qual, left_qual]]), left_bounds,
                           left_nrmac_tiff, None, True, self.extractor_info, terra_md_full, compress=True)
            self.created += 1
            self.bytes += os.path.getsize(left_nrmac_tiff)
        found_in_dest = check_file_in_dataset(connector, host, secret_key, target_dsid, left_nrmac_tiff,
                                              remove=self.overwrite)
        if not found_in_dest or self.overwrite:
            self.log_info(resource, "uploading %s" % left_nrmac_tiff)
            fileid = upload_to_dataset(connector, host, self.clowder_user, self.clowder_pass, target_dsid,
                                       left_nrmac_tiff)
            uploaded_file_ids.append(host + ("" if host.endswith("/") else "/") + "files/" + fileid)


        if not self.leftonly:
            if (not file_exists(right_nrmac_tiff) or self.overwrite):
                self.log_info(resource, "creating %s" % right_nrmac_tiff)
                create_geotiff(np.array([[right_qual, right_qual],[right_qual, right_qual]]), right_bounds,
                               right_nrmac_tiff, None, True, self.extractor_info, terra_md_full, compress=True)
                self.created += 1
                self.bytes += os.path.getsize(right_nrmac_tiff)
            found_in_dest = check_file_in_dataset(connector, host, secret_key, target_dsid, right_nrmac_tiff,
                                                  remove=self.overwrite)
            if not found_in_dest or self.overwrite:
                self.log_info(resource, "uploading %s" % right_nrmac_tiff)
                fileid = upload_to_dataset(connector, host, self.clowder_user, self.clowder_pass, target_dsid,
                                           right_nrmac_tiff)
                uploaded_file_ids.append(host + ("" if host.endswith("/") else "/") + "files/" + fileid)

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        md = {
            "files_created": uploaded_file_ids,
            "left_quality_score": left_qual
        }
        if not self.leftonly:
            md["right_quality_score"] = right_qual
        extractor_md = build_metadata(host, self.extractor_info, resource['id'], md, 'file')
        self.log_info(resource, "uploading extractor metadata to Lv1 dataset")
        remove_metadata(connector, host, secret_key, resource['id'], self.extractor_info['name'])
        upload_metadata(connector, host, secret_key, resource['id'], extractor_md)

        self.end_message(resource)


if __name__ == "__main__":
    extractor = RGB_NRMAC()
    extractor.start()
