from pathlib import Path

import acoupipe as ap

this_dir = Path(__file__).resolve().parent
src_dir = (this_dir / ".." / ".." / "src").resolve()

# -- Project information -----------------------------------------------------

project = "AcouPipe"
copyright = "Adam Kujawski, Art Pelling, Simon Jekosch, Ennes Sarradj"
author = "Adam Kujawski, Art Pelling, Simon Jekosch, Ennes Sarradj"

# The full version, including alpha/beta/rc tags
release = f"{ap.__version__}"

# -- General configuration ---------------------------------------------------

extensions = [
    "IPython.sphinxext.ipython_directive",  # Execute code during doc build
    "IPython.sphinxext.ipython_console_highlighting",  # IPython syntax highlighting
    "sphinx.ext.napoleon",  # needed to use google or numpy docstrings in python functions instead of rst
    "autoapi.extension",  # automatically create the module documentation
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",  # Link to Acoular documentation
    #"sphinx_autodoc_typehints",  #
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx_design",  # tab-set and other design elements
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
autoapi_skip_classes = ["DatasetSyntheticTestConfig", "DatasetSyntheticISM", "DatasetSyntheticISMConfig",
    "sample_rms", "sample_mic_noise_variance", "signal_seed", "DatasetSyntheticFeatureCollectionBuilder",
    "ActorHandler", "SamplerActor", "log_execution_time", "bytes_feature"]
autoapi_skip_modules = ["acoupipe.datasets.ir"]
autoapi_python_class_content = "both"
# the bibfle
bibtex_bibfiles = ["bib/refs.bib"]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_context = {
    "github_user": "adku1173",
    "github_repo": "acoupipe",
    "github_version": "master",
    "doc_path": "docs/source",
}
html_theme_options = {
    "logo": {
        "alt_text": "AcouPipe - Home",
        "text": "AcouPipe",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/adku1173/acoupipe",
            "icon": "fa-brands fa-square-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/acoupipe",
            "icon": "fa-brands fa-python",
        },
    ],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "header_links_before_dropdown": 5,
    "use_edit_page_button": True,
}
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = False
html_css_files = ["css/custom_pydata_sphinx_theme.css"]


latex_elements = {
    "preamble": r"""
\usepackage{tabular}
"""
}

# -- rst_epilog --------------------------------------------------------------

# rst_epilog is implicitly added to the end of each file before compiling to
# make the links available in all files
rst_epilog =""
# Add links.rst to rst_epilog, so external links can be used in any file
with open("contents/links.rst") as f:
     rst_epilog += f.read()

# skip certain classes
def skip_classes(app, what, name, obj, skip, options):
    if what == "module":
        skip = any([name.endswith(module_name) for module_name in autoapi_skip_modules])
    elif what == "class":
        skip = any([name.endswith(cls_name) for cls_name in autoapi_skip_classes])
    elif what == "function":
        skip = any([name.endswith(cls_name) for cls_name in autoapi_skip_classes])
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_classes)
