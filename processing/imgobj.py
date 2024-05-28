"""
Copyright 2022 by Christian König.
All rights reserved.
"""


import numpy as np
import cv2
import numba
import matplotlib.pyplot as plt


@numba.jit(nopython=True)
def neighbours(img, y, x):
    result = []
    if y-1 >= 0 and img[y-1,x]:
        result.append([y-1,x])
    if y-1 >= 0 and x-1 >= 0 and img[y-1,x-1]:
        result.append([y-1,x-1])
    if y-1 >= 0 and x+1 < len(img[0]) and img[y-1,x+1]:
        result.append([y-1,x+1])
    if y+1 < len(img) and img[y+1,x]:
        result.append([y+1,x])
    if y+1 < len(img) and x-1 >= 0 and img[y+1,x-1]:
        result.append([y+1,x-1])
    if y+1 < len(img) and x+1 < len(img[0]) and img[y+1,x+1]:
        result.append([y+1,x+1])
    if x-1 >= 0 and img[y,x-1]:
        result.append([y,x-1])
    if x+1 < len(img[0]) and img[y,x+1]:
        result.append([y,x+1])
    return result


@numba.jit(nopython=True)
def detect_opt(img, y, x, threshold_mask):
    pixel = [[y,x]]
    index = 0
    while index < len(pixel):
        y = pixel[index][0]
        x = pixel[index][1]
        if y-1 >= 0 and threshold_mask[y-1,x] and [y-1,x] not in pixel:
            pixel.append([y-1,x])
        if y+1 < len(img) and threshold_mask[y+1,x] and [y+1,x] not in pixel:
            pixel.append([y+1,x])
        if x-1 >= 0 and threshold_mask[y,x-1] and [y,x-1] not in pixel:
            pixel.append([y,x-1])
        if x+1 < len(img[0]) and threshold_mask[y,x+1] and [y,x+1] not in pixel:
            pixel.append([y,x+1])
        index += 1
    
    return pixel


def normdist(q, p0, p1):
    q  = np.array(q)
    p0 = np.array(p0)
    p1 = np.array(p1)
    n = np.linalg.norm(np.cross(p1, q-p0))
    a = np.linalg.norm(p1)
    return n/a


class ImgObj:
    
    def __init__(self, shape):
        self.mask = np.array(np.zeros(shape), dtype = np.uint8)
        
    def detect(self, img, y, x, threshold_mask):
        pixel = detect_opt(img, y, x, threshold_mask)
        m = self.mask.copy()
        for p in pixel:
            m[p[0], p[1]] = 255
        kernel = np.ones((3,3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=5).astype(bool)
        self.mask = m

    def is_at_border(self):
        """
        Returns True, if the detected object hits the border of the image.
        """
        first_row = self.mask[0, :].any()
        last_row = self.mask[-1, :].any()
        first_col = self.mask[:, 0].any()
        last_col = self.mask[:, -1].any()
        return first_row or last_row or first_col or last_col
        
    def size(self):
        return len(self.mask[self.mask])
    
    def crop(self, img):
        ind = np.argwhere(self.mask)
        y0 = ind[0][0]
        y1 = ind[-1][0]
        x0 = ind.T[1].min()
        x1 = ind.T[1].max()
        return img[y0:y1,x0:x1]
    
    def show(self, img):
        cropped = self.crop(img)
        plt.imshow(cropped)
    
    def edges(self):
        cm = self.mask
        img = np.zeros(cm.shape, dtype=np.uint8)
        contours, hierarchy = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, 255)
        return img
    
    def edge_indizes_sorted(self):
        edges = self.edges()
        ind = np.argwhere(edges)
        ind_s = [list(ind[0])]
        res = neighbours(edges, ind[0][0], ind[0][1])
        next_ = res[0]
        while next_ is not None:
            ind_s.append(next_)
            res = neighbours(edges, next_[0], next_[1])
            next_ = None
            for r in res:
                if r not in ind_s:
                    next_ = r
        return ind_s
    
    def get_snake_heads(self, step, min_dist, m_tol, distlu_tol, best=1):
        """
        experimental
        """
        ind_s = self.edge_indizes_sorted()
        snake_heads = []
        for i in range(len(ind_s)):
            p0 = ind_s[i]
            plo = p0
            puo = p0
            ml = 1e99
            mu = 0
            substep = step
            dist = 0
            distl = 0
            diff_distlu = 1e99
            while (abs(ml-mu) > m_tol or dist < min_dist or diff_distlu > distlu_tol or distl < min_dist/2) and substep < len(ind_s)/2:
                pl = ind_s[i-substep]
                pui = i+substep
                if pui >= len(ind_s):
                    pui = len(ind_s) - pui
                pu = ind_s[pui]
                dx = plo[1]-pl[1]
                if dx == 0:
                    dx = 1e-9
                ml = plo[0]-pl[0] / dx
                dx = puo[1]-pu[1]
                if dx == 0:
                    dx = 1e-9
                mu = puo[0]-pu[0] / dx
                plo = pl
                puo = pu
                substep += step
                dist = normdist(pu, pl, plo)
                distl = np.sqrt((p0[0]-pl[0])**2+(p0[1]-pl[1])**2)
                distu = np.sqrt((p0[0]-pu[0])**2+(p0[1]-pu[1])**2)
                diff_distlu = abs(distl-distu)
            
            if abs(ml-mu) <= m_tol and dist >= min_dist and diff_distlu <= distlu_tol and distl >= min_dist/2:
                snake_heads.append([p0,pl,pu,dist, ml,mu])
            
        min_dists = []
        for i in range(best):
            if len(snake_heads) == 0:
                break
            min_t = snake_heads[0]
            min_h = 0
            for h in range(len(snake_heads)):
                if snake_heads[h][3] < min_t[3]:
                    min_t = snake_heads[h]
                    min_h = h
            snake_heads.pop(min_h)
            min_dists.append(min_t)
        return min_dists
        

def obj_exists(objects, y, x):
    for obj in objects:
        if obj.mask[y,x]:
            return True
    return False