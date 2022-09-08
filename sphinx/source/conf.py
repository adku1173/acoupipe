# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../../'))


# -- Project information -----------------------------------------------------

project = 'AcouPipe'
copyright = '2021, Adam Kujawski, Art Pelling, Simon Jekosch'
author = 'Adam Kujawski, Art Pelling, Simon Jekosch'

# The full version, including alpha/beta/rc tags
release = '01.06.2021'



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary', 
    'sphinx.ext.doctest', 
    'sphinx.ext.githubpages',    
    'traits.util.trait_documenter',
    'numpydoc' #conda install -c anaconda numpydoc
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','links.rst']

# autosummary: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
autosummary_generate = True
autodoc_member_order = 'bysource'
autosummary_generate_overwrite = True # alternatively generate stub files manually with sphinx-autogen *.rst
numpydoc_show_class_members = False # Whether to show all members of a class in the Methods and Attributes sections automatically.
numpydoc_show_inherited_class_members = False #Whether to show all inherited members of a class in the Methods and Attributes sections automatically.
numpydoc_class_members_toctree = False #Whether to create a Sphinx table of contents for the lists of class methods and attributes. If a table of contents is made, Sphinx expects each entry to have a separate page.
#autodoc_mock_imports = ["acoupipe"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']



# -- rst_epilog --------------------------------------------------------------

# rst_epilog is implicitly added to the end of each file before compiling
rst_epilog =""
# Add links.rst to rst_epilog, so external links can be used in any file
with open('contents/links.rst') as f:
     rst_epilog += f.read()
