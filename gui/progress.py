"""
Copyright 2022 by Christian König.
All rights reserved.
"""

import time
from PyQt5 import QtCore


class Progress:
    
    def __init__(self, progressbar):
        """
        Progress bar helper class.
        """
        self._finished = True
        self.progressbar = progressbar
    
    def start(self, maxval):
        if self._finished:
            self._finished = False
            if self.progressbar is not None:
                self.progressbar.setValue(0)
                self.progressbar.setMaximum(maxval)
    
    def stop(self):
        self._finished = True
        self.value(self.progressbar.maximum())
    
    def inc(self, increment = 1):
        return self.value(self.progressbar.value() + increment)
    
    def value(self, curval = None):
        print(f'{curval}/{self.progressbar.maximum()}')
        if curval is not None:
            self.progressbar.setValue(curval)
            QtCore.QCoreApplication.processEvents()
            if self.progressbar.value() >= self.progressbar.maximum():
                self._finished = True
        
        return self.progressbar.value()
        
    