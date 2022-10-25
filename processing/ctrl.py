"""
Copyright 2022 by Christian König.
All rights reserved.
"""

import os
import glob
import json
import cv2

from . import preprocessing as pp
from . import helper as h
from . import scalebar as sb

import matplotlib.pyplot as plt
import numpy as np


class Ctrl:
    
    def __init__(self):
        self.is_preprocessed = False
        self.channel = -1
        self.index = -1
        self.particle_preprocessed_channel = -1
        self.particle_preprocessed_index = -1
    
    def set_directory(self, path):
        self.directory = os.path.normpath(path)
        files = glob.glob(os.path.join(self.directory, '*.czi'))
        self.files = []
        for file in files:
            self.files.append(os.path.split(file)[1])
        self.preprocessed_path = os.path.join(self.directory, 'preprocessed')
        self.metadata_path = os.path.join(self.preprocessed_path, 'metadata.json')
        if os.path.exists(self.metadata_path):
            self.is_preprocessed = True
            with open(self.metadata_path) as json_file:
                metadata = json.load(json_file)
            self.load_metadata(metadata)
        else:
            self.is_preprocessed = False
    
    def load_metadata(self, metadata):
        self.gmins = [2**16-1]*5
        self.gmaxs = [0]*5
        self.pixel_sizes = []
        self.num_channels = []
        for f in metadata.values():
            self.pixel_sizes.append(f['pixel_size'])
            self.num_channels.append(len(f['mins']))
            for i in range(len(f['mins'])):
                self.gmins[i] = min(self.gmins[i], f['mins'][i])
                self.gmaxs[i] = max(self.gmaxs[i], f['maxs'][i])
    
    def get_num_channels(self, index = None):
        if index is None:
            res = max(self.num_channels)
        else:
            self.num_channels[index]
        return res
    
    def preprocessing(self):
        os.mkdir(self.preprocessed_path)
        metadata = {}
        for file in self.files:
            res = pp.compress_data(os.path.join(self.directory, file), self.preprocessed_path)
            metadata.update(res)
        self.is_preprocessed = True
        self.load_metadata(metadata)
        
    def load_file(self, channel, index):
        file = os.path.splitext(self.files[index])[0]
        path = os.path.join(self.preprocessed_path, f'c{channel}_{file}.tiff')
        return cv2.imread(path, -1)
    
    def process_file(self, channel, index, gamma, min_, max_):
        if channel != self.channel or index != self.index:
            self.img = self.load_file(channel, index)
            self.channel = channel
            self.index = index
        img = self.img
        img = h.gamma_correction(img, gamma)
        img = h.color_scale(img, min_, max_)
        return img
    
    def merge(self, index, gammas, mins, maxs, colors):
        img = None
        for i in range(self.get_num_channels()):
            imgt = self.process_file(i, index, gammas[i], mins[i], maxs[i])
            imgt = grey16_2_rgb888(imgt, colors[i], np.uint16)
            if img is None:
                img = imgt.copy()
            else:
                img += imgt
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def add_scalebar(self, img, meter_per_pixel, length, color, line_thickness, text_size):
        return sb.add_scalebar(img, meter_per_pixel, length, color, line_thickness, text_size)
    
    def export_channel(self, channel, index, params, filename):
        img = self.process_file(channel, index, 
                                params['gamma'][channel], 
                                params['min_offset'][channel], 
                                params['max_offset'][channel])
        img = grey16_2_rgb888(img, params['channel_colors_rgb'][channel])
        if params['scalebar_en']:
            img = sb.add_scalebar(img, self.pixel_sizes[index][0], 
                                  params['scalebar_length'], 
                                  params['scalebar_color'], 
                                  params['scalebar_line_thickness'], 
                                  params['scalebar_text_size'])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)
        
    def export_merged(self, index, params, filename):
        img = self.merge(index, params['gamma'], params['min_offset'], 
                         params['max_offset'], params['channel_colors_rgb'])
        if params['scalebar_en']:
            img = sb.add_scalebar(img, self.pixel_sizes[index][0], 
                                  params['scalebar_length'], 
                                  params['scalebar_color'], 
                                  params['scalebar_line_thickness'], 
                                  params['scalebar_text_size'])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)
    
    def particle_preprocessing(self, channel, index, params):
        if self.particle_preprocessed_channel != channel or self.particle_preprocessed_index != index:
            self.particle_preprocessed_img = self.process_file(channel, index, 
                params['gamma'][channel], params['min_offset'][channel], params['max_offset'][channel])
            self.particle_preprocessed_channel = channel
            self.particle_preprocessed_index = index
            self.particles = None
        img = np.uint8(self.particle_preprocessed_img >> 8)
        
        blurred = cv2.GaussianBlur(img, (3,3),0)
        thresh = h.color_scale(blurred, params['obj_threshold'], 255) >> 8
        
        laplacian = np.abs(cv2.Laplacian(thresh,cv2.CV_64F))
        mask = laplacian > params['obj_border_threshold']
        rembor = thresh.astype(np.uint8)
        rembor[mask] = 0
        self.particle_detection_base_img = rembor
        return rembor
    
    def detect_particles(self, params):
        self.particles = h.detect_particles(self.particle_detection_base_img, 4, 0, params['obj_minsize'])
        
    def colorize_particles(self, params):
        img = grey16_2_rgb888(self.particle_preprocessed_img, 
                              params['channel_colors_rgb'][self.particle_preprocessed_channel])
        h.colorize_objects(img, self.particles, 
                           np.array(params['obj_area_color']) * 255.0,
                           np.array(params['obj_border_color']) * 255.0)
        if params['scalebar_en']:
            img = sb.add_scalebar(img, self.pixel_sizes[self.particle_preprocessed_index][0], 
                                  params['scalebar_length'], 
                                  params['scalebar_color'], 
                                  params['scalebar_line_thickness'], 
                                  params['scalebar_text_size'])
        return img
    
    def export_particles(self, channel, index, params, filename):
        self.particle_preprocessing(channel, index, params)
        if self.particles is None:
            self.detect_particles(params)
        img = self.colorize_particles(params)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img)
    
    def create_hist(self, channel, index, bins = 512):
        img = self.load_file(channel, index)
        fig, ax = plt.subplots()
        h = np.histogram(img, bins)
        ax.bar(h[1][:-1], h[0], 0.8*2**16/bins, log=True)
        # ax.xlim((self.gmins[channel], self.gmaxs[channel]))
        fig.canvas.draw()
        return np.array(fig.canvas.renderer.buffer_rgba())


def grey16_2_rgb888(img, colorf=[1,1,1], dtype = np.uint8):
    """
    16 bit grey scale image to RGB conversion

    Parameters
    ----------
    img: np,.ndarray
        input image np.uint16
    colorf: list of floats
        color channel scale factors, optional
    dtype:
        numpy data type of the returned image

    Returns
    -------
    np.ndarray of type dtype
        RGB image
    """
    rgb = np.zeros((*img.shape, 3), dtype = dtype)
    img8 = img >> 8
    rgb[:,:,0] = img8 * colorf[0]
    rgb[:,:,1] = img8 * colorf[1]
    rgb[:,:,2] = img8 * colorf[2]
    return rgb.astype(np.uint8)