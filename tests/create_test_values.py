#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create reference data for testing.
@author: kujawski
"""


import pandas as pd
from pipeline_value_test import get_pipeline
from acoupipe import WriteH5Dataset

NSAMPLES = 100

# =============================================================================
# write test_data
# =============================================================================

df = pd.DataFrame(get_pipeline(NSAMPLES,mfile="array64_d0o686.xml").get_data())
df.to_csv("test_data.csv")

# =============================================================================
# write test_data with seeds
# =============================================================================

start_seeds = [range(1,1+NSAMPLES),range(2,2+NSAMPLES),range(3,3+NSAMPLES),
               range(4,4+NSAMPLES)]

pipeline = get_pipeline(NSAMPLES,mfile="array64_d0o686.xml")
pipeline.random_seeds = start_seeds
df = pd.DataFrame(pipeline.get_data())
df.to_csv("test_data_seeds.csv")

writer = WriteH5Dataset(source=pipeline,name="test_data.h5")
writer.save()
