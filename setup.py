# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2020-2021, Adam Kujawski
#------------------------------------------------------------------------------
from setuptools import setup
from os.path import join, abspath, dirname
import os


bf_version = "21.05"
bf_author = "Adam Kujawski"

# Get the long description from the relevant file
here = abspath(dirname(__file__))
with open(join(here, 'README.rst')) as f:
    long_description = f.read()


install_requires = list([
      'acoular>=21.05',
      'setuptools',
      'ray>=1.2.0',
      'pandas',
      'h5py',
#      'parameterized',	
      'tqdm',
	])

setup_requires = list([
      'acoular>=20.02',
      'setuptools',
      'ray>=1.2.0',
      'pandas',
      'h5py',
#      'parameterized',	
      'tqdm',
	])

setup(name="acoupipe", 
      version=bf_version, 
      description="Library for sampling large-scale microphone array data with Acoular",
      long_description=long_description,
      license="BSD",
      author=bf_author,
      author_email="adam.kujawski@tu-berlin.de",
      classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Physics',
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      ],
      keywords='acoustic beamforming microphone array ',
      packages = ['acoupipe'],

      install_requires = install_requires,

      setup_requires = setup_requires,
      
      include_package_data = True,
      #to solve numba compiler 
      zip_safe=False
)

