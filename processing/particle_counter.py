"""
Copyright 2022 by Christian König.
All rights reserved.
"""

import os
import numpy as np
import cv2
import numba
import glob
import json
import sys
from openpyxl import Workbook

from . import helper as h
from . import scalebar

DIRECTORY = 'daten/6h/'

# step width in pixel for particle search
SEARCH_STEP = 2


def rgb_to_bgr(rgb):
    return np.array([rgb[2],rgb[1], rgb[0]])


def process_file_group(params, metadata):
    workbook = Workbook()
    index = 1
    for file in files:
        print('Process.. {}'.format(os.path.normpath(file)))
        
        img = cv2.imread(file)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3,3),0)
        thresh = h.color_scale(blurred, params['obj_threshold'], 255)
        clipped = np.clip(thresh, 0, thresh.max()*0.9)
        
        laplacian = cv2.Laplacian(clipped,cv2.CV_64F)
        borders=np.zeros(laplacian.shape)
        borders[laplacian>1]=laplacian[laplacian>1]
        rembor = clipped.astype(np.uint8)
        rembor[borders>0] = 0
        
        objects = h.detect_particles(rembor, SEARCH_STEP, 0, params['obj_minsize'])
        colimg = h.colorize_objects(img,
                                    objects,
                                    rgb_to_bgr(params['obj_area_color']) * 255.0,
                                    rgb_to_bgr(params['obj_border_color']) * 255.0)
        
        filename = os.path.basename(file)
        dirname = os.path.dirname(file)
        index += 1
        h.export_spreadsheet(workbook, objects, filename, index)
        scalebar.add_scalebar(colimg, metadata['pixel_size'][0],
                              params['scalebar_length'],
                              params['scalebar_color'],
                              params['scalebar_line_thickness'],
                              params['scalebar_text_size'])
        cv2.imwrite(os.path.join(dirname, 'obj_' + filename), colimg)
        
    workbook.save(filename=os.path.join(dirname, 'results.xlsx'))


if __name__ == '__main__':
    print('Particle counter')
    if len(sys.argv) == 1:
        directory = os.path.normpath(DIRECTORY)
    else:
        directory = os.path.normpath(sys.argv[1])
        
    comp_path = os.path.join(directory, os.path.split(directory)[1])
    if not os.path.exists(comp_path):
        if os.path.exists(os.path.join(directory, '__target__')):
            comp_path = directory
    
    with open(os.path.join(comp_path, 'parameters.json')) as json_file:
        parameters = json.load(json_file)
    with open(os.path.join(comp_path, 'metadata.json')) as json_file:
        metadata = json.load(json_file)
    
    for file_group in parameters.keys():
        dir_pattern = os.path.splitext(os.path.basename(file_group))[0]
        files_t = glob.glob(os.path.join(h.create_dir(comp_path, 'tiff'), dir_pattern + '.tiff'))
        files = []
        for file in files_t:
            filename = os.path.basename(file)
            if not filename.startswith('obj_'):
                files.append(file)
        process_file_group(parameters[file_group], metadata[file_group])
