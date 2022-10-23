# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:27:22 2020

@author: Christian
"""

import os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import decimal
from PIL import Image, ImageDraw, ImageFont

def to_unit_prefix(num):
    d = decimal.Decimal('{0:.0E}'.format(num))
    e = d.as_tuple().exponent
    if e < -9:
        return '{} pm'.format(int(round(num*1e12)))
    elif e < -6:
        return '{} nm'.format(int(round(num*1e9)))
    elif e < -3:
        return '{} µm'.format(int(round(num*1e6)))
    elif e < 0:
        return '{} mm'.format(int(round(num*1e3)))
    else:
        return '{} m'.format(int(round(num)))

def add_scalebar(img, meter_per_pixel, length, color = [1.0,1.0,1.0], line_thickness = 4, text_size = 80):
    if length == 0.:
        return
    cmax = np.iinfo(img.dtype).max
    color = list(np.array(color) * cmax)
    
    l = length / meter_per_pixel
    m = np.array(img.shape) * 0.9
    m2 = m.copy()
    m[1] -= l/2
    p0 = (int(m[1] - l/2), int(m[0]))
    p1 = (int(m[1] + l/2), int(m[0]))
    cv2.line(img, p0, p1, color, line_thickness)
    cv2.line(img, (p0[0], p0[1]-30), (p0[0], p0[1]+30), color, line_thickness)
    cv2.line(img, (p1[0], p1[1]-30), (p1[0], p1[1]+30), color, line_thickness)
    
    unit = to_unit_prefix(length)
    font = ImageFont.truetype("arial.ttf", text_size)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    tsize = font.getsize(unit)
    t = (m2[1] - l + (l-tsize[0])/2, m2[0])
    d.text(t, unit, fill=tuple(np.uint32(color).tolist()), font=font)
    img = np.array(im)
    return img

if __name__ == '__main__':
    IMG_PATH = "D:\\Projekte\\microscopy\\examples\\corona\\tiff\\2020-04-27_Chip-2_HUVEC_40x2.tiff"
    
    img = cv2.imread(IMG_PATH)
    imsb = add_scalebar(img, 5e-7, 150e-6, text_size = 80)
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image', imsb)
    cv2.resizeWindow('image', 1024,768)
    cv2.waitKey(0)  
    cv2.destroyAllWindows() 