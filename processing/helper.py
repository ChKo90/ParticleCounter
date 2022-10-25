"""
Copyright 2022 by Christian König.
All rights reserved.
"""

import numpy as np
import cv2
import numba
import os

from .imgobj import ImgObj
from .imgobj import obj_exists


def load(files):
    dfiles = {}
    for file in files:
        img = cv2.imread(file, -1)
        ch, name = os.path.basename(file).split('_', 1)
        ch = int(ch[1:])
        if name not in dfiles:
            dfiles[name] = [None]*5
        dfiles[name][ch] = img
    
    images = []
    for val in dfiles.values():
        images.append([])
        for im in val:
            if im is not None:
                images[-1].append(im)
    return np.array(images, dtype = img.dtype), list(dfiles.keys())


@numba.jit(nopython=True)
def gamma_correction(img, gamma):
    imin, imax = 0, 2**16-1
    img_c = img.copy()
    img_c = ((img_c - imin) / (imax - imin)) ** gamma
    img_c = img_c * (imax - imin) + imin
    img_c += 0.5
    return img_c.astype(np.uint16)


@numba.jit(nopython=True)
def color_scale_fast(img, min_, max_):
    m = (2**16-1)/(max_-min_)
    img = m*img-m*min_
    img += 0.5
    return img.astype(np.uint16)


def color_scale(img, min_, max_):
    img = np.clip(img, min_, max_)
    return color_scale_fast(img, min_, max_)


def detect_particles(img, step, threshold, minsize):
    objects = []
    for y in range(0, len(img), step):
        for x in range(0, len(img[0]), step):
            if img[y][x] > threshold:
                if not obj_exists(objects, y, x):
                    obj = ImgObj(img.shape)
                    if obj.detect(img, y, x, threshold, minsize):
                        objects.append(obj)
                        print('Object {} at row {} column {} with size {}'.format(len(objects)-1, y, x, obj.size()))
    return objects


# palette
coltbl = [[0,0,32],[0,32,0],[32,0,0],
          [0,0,64],[0,64,0],[64,0,0],
          [0,0,96],[0,96,0],[96,0,0],
          [0,0,128],[0,128,0],[128,0,0],
          [0,0,160],[0,160,0],[160,0,0],
          [0,0,192],[0,192,0],[192,0,0],
          [0,0,255],[0,255,0],[255,0,0]]


def colorize_objects(img, objects, objcol = [255,255,255], bordercol = [0,0,0]):
    sum_overlay = np.zeros(img.shape[0:2], dtype=np.uint8)
    sum_edges = np.zeros(img.shape[0:2], dtype=np.uint8)
    for obj in objects:
        sum_edges   += obj.edges()
        sum_overlay += obj.mask.astype(np.uint8)

    overlay = cv2.cvtColor(sum_overlay, cv2.COLOR_GRAY2BGR)*objcol
    overlay[sum_edges>0] = bordercol
    cv2.addWeighted(overlay.astype(img.dtype), 0.8, img, 0.2, 0, img)
    return img


def create_dir(directory, subdir):
    result_path = os.path.join(directory, subdir)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    return result_path


def export_spreadsheet(workbook, objects, name, index):
    area = 0
    for obj in objects:
        area += obj.size()
    
    sheet = workbook.active
    
    sheet["A1"] = "name"
    sheet["A2"] = "sum object area"
    sheet["A3"] = "number of detected objects"
    sheet["A4"] = "correction"
    sheet["A5"] = ""
    sheet["A6"] = "average"
    
    sheet.cell(1, index).value = name
    sheet.cell(2, index).value = area
    carea = sheet.cell(2, index).coordinate
    sheet.cell(3, index).value = len(objects)
    cobjects = sheet.cell(3, index).coordinate
    sheet.cell(4, index).value = 0
    ccorr = sheet.cell(4, index).coordinate
    sheet.cell(5, index).value = ""
    sheet.cell(6, index).value = "={}/({}+{})".format(carea, cobjects, ccorr)


def rgb_to_bgr(rgb):
    return np.array([rgb[2],rgb[1], rgb[0]])
