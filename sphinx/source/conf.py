# Configuration file for the Sphinx documentation builder.
#
# -- Path setup --------------------------------------------------------------
from pathlib import Path

this_dir = Path(__file__).resolve().parent
src_dir = (this_dir / ".." / ".." / "src").resolve()

# -- Project information -----------------------------------------------------

project = "AcouPipe"
copyright = "Adam Kujawski, Art Pelling, Simon Jekosch, Ennes Sarradj"
author = "Adam Kujawski, Art Pelling, Simon Jekosch, Ennes Sarradj"

# The full version, including alpha/beta/rc tags
release = "30.09.2023"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",  # needed to use google or numpy docstrings in python functions instead of rst
    "autoapi.extension",  # automatically create the module documentation
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",  # Link to Acoular documentation
    #"sphinx_autodoc_typehints",  #
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "traits.util.trait_documenter",
    #"numpydoc", #conda install -c anaconda numpydoc
    "nbsphinx", # allows to include jupyter notebooks into rst documentation
    "sphinxcontrib.bibtex", # to cite papers if necessary
]

# auto api configuration
autoapi_type = "python"
autoapi_dirs = [src_dir / "acoupipe"]
autoapi_add_toctree_entry = False  # no seperate index.rst file created by autoapi
autoapi_options = ["show-inheritance"]
autoapi_skip_classes = ["Dataset1TestConfig", "sample_rms", "sample_mic_noise_variance",
    "signal_seed", "Dataset1FeatureCollectionBuilder"]
autoapi_python_class_content = "both"
# the bibfle
bibtex_bibfiles = ["bib/refs.bib"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- rst_epilog --------------------------------------------------------------

# rst_epilog is implicitly added to the end of each file before compiling to
# make the links available in all files
rst_epilog =""
# Add links.rst to rst_epilog, so external links can be used in any file
with open("contents/links.rst") as f:
     rst_epilog += f.read()

# skip certain classes
def skip_classes(app, what, name, obj, skip, options):
    if what == "class":
        skip = any([name.endswith(cls_name) for cls_name in autoapi_skip_classes])
    elif what == "function":
        skip = any([name.endswith(cls_name) for cls_name in autoapi_skip_classes])
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_classes)
