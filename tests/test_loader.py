#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:26:28 2021

@author: Jekosch
"""

import unittest
from acoupipe import BaseLoadDataset, LoadH5Dataset

LOADER_CLASSES = [BaseLoadDataset, LoadH5Dataset]

class Test_Loader(unittest.TestCase):

    def test_instancing(self):
        """create an instance of each class defined in module"""
        for c in LOADER_CLASSES:
            c()    
            
    def test_h5_data(self):
        ds =  LoadH5Dataset(name='test_data.h5')
        
        self.assertEqual(ds.numsamples,100)
        self.assertEqual(ds.numfeatures,6)
        self.assertEqual(ds.basename,'test_data')
        ds.load_dataset()
        self.assertEqual(ds.dataset['1']['rms_sources'][()][0],5.25081601669544)
        
        


    
