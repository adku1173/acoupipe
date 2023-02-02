import os
import tempfile
import unittest

from pipeline_value_test import get_pipeline

from acoupipe.loader import LoadH5Dataset
from acoupipe.writer import WriteH5Dataset


class TestWriteAndLoad(unittest.TestCase):

    def setUp(self):
        pipeline = get_pipeline(5)
        pipeline.random_seeds = {1:range(1,1+5),2:range(2,2+5),3:range(3,3+5),
                    4:range(4,4+5)}
        self.name = os.path.join(tempfile.mkdtemp(),"test_data.h5")
        writer = WriteH5Dataset(source=pipeline,name=self.name)
        writer.save()

    def test_load_h5_data(self):
        ds =  LoadH5Dataset(name=self.name)
        ds.load_dataset()
        self.assertEqual(ds.numsamples,5)
        self.assertEqual(ds.basename,"test_data")
        ds.load_dataset()
        self.assertEqual(ds.dataset["1"]["data"][()],True)
        
        


    
