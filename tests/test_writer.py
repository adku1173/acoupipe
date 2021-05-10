"""
Created on Tue Jan 19 16:40:55 2021

"""

import unittest


from acoupipe import BaseWriteDataset, WriteH5Dataset

from pipeline_value_test import get_pipeline


WRITER_CLASSES = [BaseWriteDataset, WriteH5Dataset]


class Test_Writer(unittest.TestCase):

    def test_instancing(self):
        """create an instance of each class defined in module"""
        for c in WRITER_CLASSES:
            c()    
    
    def test_pipeline(self):
        """create an instance of each class defined in module"""
        for c in WRITER_CLASSES:
            c.source = get_pipeline
            
    
