# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:54:31 2021

@author: Christian
"""

from PyQt5 import QtWidgets, QtGui, uic
from PyQt5.QtCore import Qt
import numpy as np
import cv2
import sys
import os
import shutil
import threading
import glob
import json
from datetime import datetime
from openpyxl import Workbook

import ctrl
from progress import Progress
import helper as h

DEFAULT_PARAMS = {  'scalebar_en' : True,
                    'scalebar_length' : 100e-6,
                    'scalebar_color' : [1.0, 1.0, 1.0],
                    'scalebar_line_thickness' : 4,
                    'scalebar_text_size' : 80,
                    'gamma' : [1.0]*5,
                    'min_offset' : [0]*5,
                    'max_offset' : [0]*5,
                    'channel_colors_rgb' : [[0,1,0],[0,0,1],[1,0,0],[1,1,0],[0,1,1]],
                    'obj_threshold' : 0,
                    'obj_border_threshold' : 255,
                    'obj_minsize' : 20,
                    'obj_maxsize' : -1,
                    'obj_border_color' : [1,0,0],
                    'obj_area_color' : [1,1,1]
                    }

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('mainwindow.ui', self) # Load the .ui file
        
        self.ctrl = ctrl.Ctrl()
        self.params = dict(DEFAULT_PARAMS)
        self.ch = 0
        self.mode = 0
        
        self.skip_gui_update = False
        
        self.btn_open_directory.clicked.connect(self.open_directory)
        self.btn_preprocessing.clicked.connect(self.preprocessing)
        self.list_files.currentRowChanged.connect(self.file_changed)
        self.cb_channel.currentIndexChanged.connect(self.channel_changed)
        self.scrollArea_preview.resizeEvent = self.preview_resize
        
        self.btn_channel_color.clicked.connect(self.channel_color)
        self.btn_sb_color.clicked.connect(self.scalebar_color)
        
        self.btn_merged.toggled.connect(self.value_changed)
        
        self.btn_params_add.clicked.connect(self.add_params)
        self.btn_params_save.clicked.connect(self.save_params)
        self.btn_params_delete.clicked.connect(self.delete_params)
        self.list_params.currentItemChanged.connect(self.params_changed)
        self.list_params.itemChanged.connect(self.params_renamed)
        
        self.btn_export_directory.clicked.connect(self.export_directory)
        self.btn_export_current.clicked.connect(self.export_current)
        self.btn_export_all.clicked.connect(self.export_all)
        
        self.btn_obj_area_color.clicked.connect(self.obj_area_color)
        self.btn_obj_border_color.clicked.connect(self.obj_border_color)
        self.btn_obj_detection.clicked.connect(self.obj_detection)
        self.btn_show_detection.toggled.connect(self.value_changed)
        
        self.tabs_config.currentChanged.connect(self.mode_changed)
        
        # progress bar
        self.progress_bar = QtWidgets.QProgressBar(self.statusbar)
        self.statusbar.addPermanentWidget(self.progress_bar, 1)
        self.progress_bar.setTextVisible(False)
        self.progress = Progress(self.progress_bar)
                
        self.toolBox_progress.setCurrentIndex(0)
        self.tabs_config.setCurrentIndex(0)
        self.btn_preprocessing.setEnabled(False)
        self.tabs_config.setEnabled(False)
        self.page_params.setEnabled(False)
        self.page_export.setEnabled(False)
        self.btn_show_detection.setVisible(False)
        self.show() # Show the GUI
        
        self.thr = threading.Thread(target=self.update_thread)
        self.update_event = threading.Event()
        self.thr.start()
        
    def open_directory(self):
        dlg = QtWidgets.QFileDialog(self, 'Select a directory!')
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.DirectoryOnly)
        if dlg.exec_():
            self.list_files.clear()
            directory = dlg.selectedFiles()[0]
            print('Directory ' + directory)
            self.ctrl.set_directory(directory)
            self.edit_directory.setText(directory)
            self.list_files.addItems(self.ctrl.files)
            self.load_directory_data()
            self.load_params()
            self.btn_preprocessing.setEnabled(True)
            self.tabs_config.setEnabled(self.ctrl.is_preprocessed)
            self.page_params.setEnabled(self.ctrl.is_preprocessed)
            self.page_export.setEnabled(self.ctrl.is_preprocessed)
            export_dir = os.path.join(self.ctrl.directory, 'export')
            self.edit_export_directory.setText(export_dir)
            if not os.path.exists(export_dir):
                os.mkdir(export_dir)
            
    def preprocessing(self):
        if os.path.exists(self.ctrl.preprocessed_path):
            shutil.rmtree(self.ctrl.preprocessed_path)
        self.ctrl.preprocessing()
        self.load_directory_data()
        self.tabs_config.setEnabled(self.ctrl.is_preprocessed)
        self.page_params.setEnabled(self.ctrl.is_preprocessed)
        self.page_export.setEnabled(self.ctrl.is_preprocessed)
        
    def load_directory_data(self):
        print('load_directory_data')
        self.skip_gui_update = True
        self.params = dict(DEFAULT_PARAMS)
        self.cb_channel.clear()
        self.edit_gmax.setText('')
        self.edit_gmin.setText('')
        if self.ctrl.is_preprocessed:
            ch = self.ctrl.get_num_channels()
            for i in range(ch):
                self.cb_channel.addItem(str(i))
            self.cb_channel.setCurrentIndex(0)
            self.ch = 0
            self.edit_gmax.setText(str(self.ctrl.gmaxs[:ch]))
            self.edit_gmin.setText(str(self.ctrl.gmins[:ch]))
            self.params['min_offset'] = self.ctrl.gmins
            self.params['max_offset'] = self.ctrl.gmaxs
            
        self.update_gui_values()
        self.list_files.setCurrentRow(0)
        
    def file_changed(self, row):
        self.btn_show_detection.setChecked(False)
        self.btn_show_detection.setEnabled(False)
        self.update_all(None)
        
    def update_all(self, row):
        if not self.ctrl.is_preprocessed or self.skip_gui_update:
            self.skip_gui_update = False
            return
        print('File Changed')
        if self.thr.is_alive():
            self.update_event.set()
    
    def mode_preview_preprocessed(self, file_row):
        if self.btn_merged.isChecked():
            img = self.ctrl.merge(file_row, self.params['gamma'], 
                                  self.params['min_offset'], 
                                  self.params['max_offset'], 
                                  self.params['channel_colors_rgb'])
        else:
            img = self.ctrl.process_file(self.ch, file_row, self.params['gamma'][self.ch],
                                         self.params['min_offset'][self.ch], 
                                         self.params['max_offset'][self.ch])
            img = ctrl.grey16_2_rgb888(img, self.params['channel_colors_rgb'][self.ch])
            
        if self.gb_sb.isChecked():
            img = self.ctrl.add_scalebar(img, self.ctrl.pixel_sizes[self.list_files.currentRow()][0],
                                              self.spin_sb_length.value()/1e6,
                                              self.params['scalebar_color'],
                                              self.spin_sb_thickness.value(),
                                              self.spin_sb_text_size.value())
        
        # preview
        image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 
                             img.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
        return image
    
    def mode_preview_particle(self, file_row):
        if self.btn_show_detection.isChecked():
            img = self.ctrl.colorize_particles(self.params)
            image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 
                                 img.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
        else:
            img = self.ctrl.particle_preprocessing(self.ch, file_row, self.params)
            image = QtGui.QImage(img.data, img.shape[1], img.shape[0], 
                                 img.shape[1], QtGui.QImage.Format.Format_Grayscale8)
        return image
    
    def update_thread(self):
        print('Thread running')
        while self.isVisible():
            if self.update_event.wait(0.5):
                self.update_event.clear()
                print('update')
                row = self.list_files.currentRow()
                if self.mode == 0:
                    image = self.mode_preview_preprocessed(row)
                elif self.mode == 1:
                    image = self.mode_preview_particle(row)
                
                pix = QtGui.QPixmap.fromImage(image)
                self.label_preview.setPixmap(pix)
                self.edit_pixel_size.setText(str(self.ctrl.pixel_sizes[self.list_files.currentRow()]))
                
                # histogram
                # h = self.ctrl.create_hist(self.ch, row)
                # imageh = QtGui.QImage(h.data, h.shape[1], h.shape[0], 
                #                       h.shape[1] * 4, QtGui.QImage.Format.Format_RGBA8888)
                # pix = QtGui.QPixmap.fromImage(imageh)
                # self.label_hist.setPixmap(pix)
                
                size = self.scrollAreaWidget_preview.size()
                self.preview_resize(QtGui.QResizeEvent(size, size))
        
    def update_gui_values(self):
        try:
            self.spin_gamma.valueChanged.disconnect()
            self.spin_min.valueChanged.disconnect()
            self.spin_max.valueChanged.disconnect()
            self.spin_sb_length.valueChanged.disconnect()
            self.spin_sb_thickness.valueChanged.disconnect()
            self.spin_sb_text_size.valueChanged.disconnect()
            self.gb_sb.toggled.disconnect()
            self.spin_obj_pixel_threshold.valueChanged.disconnect()
            self.spin_obj_border_threshold.valueChanged.disconnect()
            self.spin_obj_minsize.valueChanged.disconnect()
            self.spin_obj_maxsize.valueChanged.disconnect()
        except:
            pass
        
        print('update GUI params')
        self.spin_gamma.setValue(self.params['gamma'][self.ch])
        self.spin_min.setValue(self.params['min_offset'][self.ch])
        self.spin_max.setValue(self.params['max_offset'][self.ch])
        self.spin_sb_length.setValue(int(self.params['scalebar_length'] * 1e6))
        self.spin_sb_thickness.setValue(self.params['scalebar_line_thickness'])
        self.spin_sb_text_size.setValue(self.params['scalebar_text_size'])
        self.gb_sb.setChecked(self.params['scalebar_en'])
        self.spin_obj_pixel_threshold.setValue(self.params['obj_threshold'])
        self.spin_obj_border_threshold.setValue(self.params['obj_border_threshold'])
        self.spin_obj_minsize.setValue(self.params['obj_minsize'])
        self.spin_obj_maxsize.setValue(self.params['obj_maxsize'])
        
        self.spin_gamma.valueChanged.connect(self.value_changed)
        self.spin_min.valueChanged.connect(self.value_changed)
        self.spin_max.valueChanged.connect(self.value_changed)
        self.spin_sb_length.valueChanged.connect(self.value_changed)
        self.spin_sb_thickness.valueChanged.connect(self.value_changed)
        self.spin_sb_text_size.valueChanged.connect(self.value_changed)
        self.gb_sb.toggled.connect(self.value_changed)
        self.spin_obj_pixel_threshold.valueChanged.connect(self.value_changed)
        self.spin_obj_border_threshold.valueChanged.connect(self.value_changed)
        self.spin_obj_minsize.valueChanged.connect(self.value_changed)
        self.spin_obj_maxsize.valueChanged.connect(self.value_changed)
        
    def value_changed(self, val=None):
        print(f'Value Changed: {val}')
        self.params['gamma'][self.ch] = self.spin_gamma.value()
        self.params['min_offset'][self.ch] = self.spin_min.value()
        self.params['max_offset'][self.ch] = self.spin_max.value()
        self.params['scalebar_length'] = self.spin_sb_length.value() / 1e6
        self.params['scalebar_line_thickness'] = self.spin_sb_thickness.value()
        self.params['scalebar_text_size'] = self.spin_sb_text_size.value()
        self.params['scalebar_en'] = self.gb_sb.isChecked()
        self.params['obj_threshold'] = self.spin_obj_pixel_threshold.value()
        self.params['obj_border_threshold'] = self.spin_obj_border_threshold.value()
        self.params['obj_minsize'] = self.spin_obj_minsize.value()
        self.params['obj_maxsize'] = self.spin_obj_maxsize.value()
        self.update_all(self.list_files.currentRow())
        
    def channel_changed(self, val = None):
        try:
            self.ch = int(self.cb_channel.currentText())
        except:
            pass
        self.update_gui_values()
        self.update_all(self.list_files.currentRow())
        
    def preview_resize(self, ev):
        if self.label_preview.pixmap() is not None and ev.size() == self.label_preview.pixmap().size():
            return
        print('preview_resize')
        QtWidgets.QScrollArea.resizeEvent(self.scrollArea_preview, ev)
        if self.label_preview.pixmap() is not None:
            size = self.label_preview.pixmap().size()
            size.scale(ev.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.scrollAreaWidget_preview.resize(size)
    
    def channel_color(self):
        color = QtGui.QColor(*np.uint32(np.array(self.params['channel_colors_rgb'][self.ch]) * 255).tolist())
        dlg = QtWidgets.QColorDialog(color, self)
        if dlg.exec_():
            col = dlg.currentColor()
            self.params['channel_colors_rgb'][self.ch] = [col.redF(), col.greenF(), col.blueF()]
            self.value_changed()
            
    def scalebar_color(self):
        color = QtGui.QColor(*np.uint32(np.array(self.params['scalebar_color']) * 255).tolist())
        dlg = QtWidgets.QColorDialog(color, self)
        if dlg.exec_():
            col = dlg.currentColor()
            self.params['scalebar_color'] = [col.redF(), col.greenF(), col.blueF()]
            self.value_changed()
    
    def add_parameter_set(self, name):
        item = QtWidgets.QListWidgetItem(name, self.list_params)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        return item
    
    def load_params(self):
        self.path_parameters = os.path.join(self.ctrl.directory, 'parameters')
        files = glob.glob(os.path.join(self.path_parameters, '*.json'))
        self.parameter_sets = []
        self.list_params.clear()
        for file in files:
            name = os.path.splitext(os.path.basename(file))[0]
            self.parameter_sets.append(name)
            self.add_parameter_set(name)
        self.parameter_sets.sort()
        self.list_params.setCurrentRow(self.list_params.count()-1)
    
    def add_params(self):
        self.list_params.itemChanged.disconnect()
        if not os.path.exists(self.path_parameters):
            os.mkdir(self.path_parameters)
        name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        name_cor = str(name)
        i = 1
        while name_cor in self.parameter_sets:
            name_cor = name + f'_{i}'
            i += 1
        self.parameter_sets.append(name_cor)
        self.parameter_sets.sort()
        item = self.add_parameter_set(name_cor)
        with open(os.path.join(self.path_parameters, name_cor + '.json'), 'w') as file:
            json.dump(self.params, file, indent=4)
        self.list_params.setCurrentItem(item)
        self.list_params.itemChanged.connect(self.params_renamed)
    
    def save_params(self):
        if self.list_params.currentRow() > -1:
            name = self.list_params.currentItem().text()
            with open(os.path.join(self.path_parameters, name + '.json'), 'w') as file:
                json.dump(self.params, file, indent=4)
    
    def delete_params(self):
        row = self.list_params.currentRow()
        if row > -1:
            name = self.list_params.currentItem().text()
            file = os.path.join(self.path_parameters, 
                                name + '.json')
            if os.path.exists(file):
                os.remove(file)
            self.list_params.takeItem(row)
            self.parameter_sets.remove(name)
            self.parameter_sets.sort()
            
    def params_changed(self, current_item, previous_item):
        if current_item is not None:
            print('params_changed')
            file = os.path.join(self.path_parameters, current_item.text() + '.json')
            with open(file) as json_file:
                self.params = dict(DEFAULT_PARAMS)
                self.params.update(json.load(json_file))
            self.update_gui_values()
            self.value_changed()
            
    def params_renamed(self, item):
        row = self.list_params.currentRow()
        old = self.parameter_sets[row]
        if old != item.text():
            print('params_renamed')
            os.rename(os.path.join(self.path_parameters, old + '.json'),
                      os.path.join(self.path_parameters, item.text() + '.json'))
            self.parameter_sets[row] = item.text()
            self.parameter_sets.sort()
    
    def export_directory(self):
        dlg = QtWidgets.QFileDialog(self, 'Select a export directory!')
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.DirectoryOnly)
        if dlg.exec_():
            directory = dlg.selectedFiles()[0]
            self.edit_export_directory.setText(directory)
    
    def export(self, index, workbook = None):
        basename = os.path.splitext(self.ctrl.files[index])[0] + self.cb_export_format.currentText()
        if self.tabs_config.currentIndex() == 0:
            if self.check_export_channels.checkState():
                for i in range(self.ctrl.get_num_channels()):
                    self.progress.inc()
                    filename = os.path.join(self.edit_export_directory.text(), f'c{i}_{basename}')
                    self.ctrl.export_channel(i, index, self.params, filename)
                    print(f'Export channel {i}: {filename}')
            if self.check_export_merged.checkState():
                self.progress.inc()
                filename = os.path.join(self.edit_export_directory.text(), basename)
                self.ctrl.export_merged(index, self.params, filename)
                print(f'Export merged: {filename}')
        elif self.tabs_config.currentIndex() == 1:
            self.progress.inc()
            filename = os.path.join(self.edit_export_directory.text(), f'particles_c{self.ch}_{basename}')
            self.ctrl.export_particles(self.ch, index, self.params, filename)
            if workbook is not None:
                h.export_spreadsheet(workbook, self.ctrl.particles, basename, index + 1)
            print(f'Export particles channel {self.ch}: {filename}')
    
    def _long_processing_start(self, maxval):
        self.progress.start(maxval)
        self.setEnabled(False)
        
    def _long_processing_stop(self):
        self.progress.stop()
        self.setEnabled(True)
    
    def export_current(self):
        if self.tabs_config.currentIndex() == 0:
            self._long_processing_start(self.ctrl.get_num_channels() + int(self.check_export_merged.isChecked()))
        elif self.tabs_config.currentIndex() == 1:
            self._long_processing_start(1)
        self.export(self.list_files.currentRow())
        self._long_processing_stop()
    
    def export_all(self):
        if self.tabs_config.currentIndex() == 0:
            self._long_processing_start(len(self.ctrl.files) *
                                        (self.ctrl.get_num_channels() + int(self.check_export_merged.isChecked())))
        elif self.tabs_config.currentIndex() == 1:
            self._long_processing_start(len(self.ctrl.files))
            workbook = Workbook()
        for i in range(len(self.ctrl.files)):
            self.export(i, workbook)
        if self.tabs_config.currentIndex() == 1:
            workbook.save(filename=os.path.join(self.edit_export_directory.text(), 'results.xlsx'))
        self._long_processing_stop()
            
    def mode_changed(self, index):
        self.mode = index
        if index == 0:
            self.gb_export_processed.setEnabled(True)
            self.btn_merged.setVisible(True)
            self.btn_show_detection.setVisible(False)
        elif index == 1:
            self.gb_export_processed.setEnabled(False)
            self.btn_merged.setVisible(False)
            self.btn_show_detection.setVisible(True)
        self.value_changed()
        
    def obj_area_color(self):
        color = QtGui.QColor(*np.uint32(np.array(self.params['obj_area_color']) * 255).tolist())
        dlg = QtWidgets.QColorDialog(color, self)
        if dlg.exec_():
            col = dlg.currentColor()
            self.params['obj_area_color'] = [col.redF(), col.greenF(), col.blueF()]
            self.value_changed()
    
    def obj_border_color(self):
        color = QtGui.QColor(*np.uint32(np.array(self.params['obj_border_color']) * 255).tolist())
        dlg = QtWidgets.QColorDialog(color, self)
        if dlg.exec_():
            col = dlg.currentColor()
            self.params['obj_border_color'] = [col.redF(), col.greenF(), col.blueF()]
            self.value_changed()
            
    def obj_detection(self):
        self.ctrl.detect_particles(self.params)
        self.btn_show_detection.setEnabled(True)
        self.btn_show_detection.setChecked(True)
    
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = Ui() # Create an instance of our class
    if 'get_ipython' in globals():
        window.show()
    else:
        sys.exit(app.exec_()) # Start the application