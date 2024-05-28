"""
Copyright 2022 by Christian König.
All rights reserved.
"""

import cv2
import numpy as np
import decimal
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pilversion


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
    if int(pilversion.split('.')[0]) > 9:
        left, top, right, bottom = font.getbbox(unit)
        tsize = right - left
    else:
        tsize = font.getsize(unit)[0]
    t = (m2[1] - l + (l-tsize)/2, m2[0])
    d.text(t, unit, fill=tuple(np.uint32(color).tolist()), font=font)
    img = np.array(im)
    return img

