# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:52:08 2020

@author: Christian
"""

import sys
import os
import czifile
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import json
import numba

from helper import *
from scalebar import *

DIRECTORY = 'examples/6h/'
MAX_CHANNELS = 5
HIST_BINS = 512

def compress_data(file, comp_path):
    czi = czifile.CziFile(file)
    d_tmp = czi.metadata(False)['ImageDocument']['Metadata']['Scaling']['Items']['Distance']
    pixel_size = [d_tmp[0]['Value'], d_tmp[1]['Value'], d_tmp[2]['Value']]
    inp = czi.asarray(-1)
    czi.close()
    
    inp = inp[0][:,:,:,:,0]
    filename = os.path.split(file)[1]
    newfilename = os.path.splitext(filename)[0] + '.tiff'
    mins = []
    maxs = []
    for i,ch in enumerate(inp):
        img=np.amax(ch,0)
        mins.append(int(img.min()))
        maxs.append(int(img.max()))
        outfilename = 'c{}_{}'.format(i, newfilename)
        print('Save merged and compressed file {}'.format(outfilename))
        cv2.imwrite(os.path.join(comp_path, outfilename), img)
    
    res = {'pixel_size'    : pixel_size,
           'mins'          : mins,
           'maxs'          : maxs
           }
    out = {filename : res}
    
    metafile, metadata = check_metadata(comp_path)
    metadata.update(out)
    with open(metafile, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    return out

def check_parameters(path):
    filepath = os.path.join(path, 'parameters.json')
    if os.path.exists(filepath):
        with open(filepath) as json_file:
            parameters = json.load(json_file)
    else:
        parameters = {'*.tiff' : 
                          {'scalebar_length' : 50e-6,
                           'scalebar_color' : [1.0, 1.0, 1.0],
                           'scalebar_line_thickness' : 4,
                           'scalebar_text_size' : 80,
                           'gamma' : [1.0]*5,
                           'min_offset' : [0]*5,
                           'max_offset' : [0]*5,
                           'channel_colors_rgb' : [[0,1,0],[0,0,1],[1,0,0],[0,0,0],[0,0,0]],
                           'channel_outputs' : [],
                           'merged_channel_outputs' : ['tiff'],
                           'obj_threshold' : 20,
                           'obj_minsize' : 20,
                           'obj_border_color' : [1,0,0],
                           'obj_area_color' : [1,1,1]
                           }
                      }
        with open(filepath, 'w') as json_file:
            json.dump(parameters, json_file, indent=4)
    return parameters

def check_metadata(directory):
    metafile = os.path.join(directory, 'metadata.json')
    metadata = {}
    if os.path.exists(metafile):
        with open(metafile) as json_file:
            metadata = json.load(json_file)
    return metafile, metadata

def create_hists(imgs, minmins, maxmaxs):
    for ch in range(len(imgs[0])):
        filename = os.path.join(comp_path, 'c{}_hist.png'.format(ch))
        plt.figure()
        h = np.histogram(imgs[:,ch], HIST_BINS)
        plt.bar(h[1][:-1], h[0], 0.8*2**16/HIST_BINS, log=True)
        plt.xlim((minmins[ch], maxmaxs[ch]))
        plt.savefig(filename)
        plt.close()

def get_global_pixel_ranges(imgs, metadata):
    print('Determine global pixel ranges...')
    for ch in range(len(imgs[0])):
        metadata['minmins'][ch] = int(imgs[:,ch,:,:].min())
        metadata['maxmaxs'][ch] = int(imgs[:,ch,:,:].max())

def process_channels(inp, gammas, minmins, maxmaxs):
    for i in range(len(inp)):
        inp[i] = gamma_correction(inp[i], gammas[i])
        inp[i] = color_scale(inp[i], minmins[i], maxmaxs[i])
    return inp

def process_file_group(imgs, filenames, params, metadata, determine_minmax):
    if determine_minmax:
        get_global_pixel_ranges(imgs, metadata)
    minmins = metadata['minmins']
    maxmaxs = metadata['maxmaxs']
    print('Global min/max pixel values of each channel: {}/{}'.format(minmins, maxmaxs))
    
    if determine_minmax:
        create_hists(imgs, minmins, maxmaxs)
    
    for img, filename in zip(imgs, filenames):
        out = process_channels(img, 
                               params['gamma'], 
                               np.add(minmins, params['min_offset']),
                               np.add(maxmaxs, params['max_offset']))
            
        rgbs = []
        merged = np.zeros((*out[0].shape, 3), dtype = np.uint16)
        for channel in range(len(out)):
            rgb = np.zeros((*out[channel].shape, 3), dtype = np.uint16)
            rgb[:,:,0] = out[channel] * float(params['channel_colors_rgb'][channel][2])
            rgb[:,:,1] = out[channel] * float(params['channel_colors_rgb'][channel][1])
            rgb[:,:,2] = out[channel] * float(params['channel_colors_rgb'][channel][0])
            rgbs.append(rgb)
            merged += rgb
        
        for channel in range(len(rgbs)):
            for file_format in params['channel_outputs']:
                add_scalebar(rgbs[channel], metadata['pixel_size'][0], 
                             params['scalebar_length'], 
                             params['scalebar_color'], 
                             params['scalebar_line_thickness'],
                             params['scalebar_text_size'])
                cv2.imwrite(os.path.join(create_dir(comp_path, file_format), 'c{}_{}.{}'.format(channel, filename, file_format)), rgbs[channel])
        
        for file_format in params['merged_channel_outputs']:
            add_scalebar(merged, metadata['pixel_size'][0], 
                         params['scalebar_length'], 
                         params['scalebar_color'], 
                         params['scalebar_line_thickness'],
                         params['scalebar_text_size'])
            cv2.imwrite(os.path.join(create_dir(comp_path, file_format), '{}.{}'.format(filename, file_format)), merged)
            

if __name__ == '__main__':
    if len(sys.argv) == 1:
        directory = os.path.normpath(DIRECTORY)
    else:
        directory = os.path.normpath(sys.argv[1])
        
    comp_path = os.path.join(directory, os.path.split(directory)[1])
    if not os.path.exists(comp_path):
        if os.path.exists(os.path.join(directory, '__target__')):
            comp_path = directory
        else:
            os.mkdir(comp_path)
            pixel_size = compress_data(directory, comp_path)
    
    parameters = check_parameters(comp_path)
    metafile, metadata = check_metadata(comp_path)
    
    for file_group in parameters.keys():
        if file_group not in metadata:
            metadata[file_group] = {'pixel_size' : pixel_size,
                                    'minmins' : [0]*5, 
                                    'maxmaxs' : [2**16-1]*5}
            determine_minmax = True
        else:
            determine_minmax = False
        files = glob.glob(os.path.join(comp_path, file_group))
        imgs, filenames = load(files)
        process_file_group(imgs, filenames, parameters[file_group], metadata[file_group], determine_minmax)
        
    with open(metafile, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    print('Image processing finished successfully!!')
    