from pathlib import Path

import acoupipe as ap

from acoular_sphinx import (
    PACKAGE_FRAME_EXTENSIONS,
    build_github_context,
    build_html_context,
    configure_package_theme_options,
    resolve_docs_build_config,
)

this_dir = Path(__file__).resolve().parent
src_dir = (this_dir / ".." / ".." / "src").resolve()

# -- Project information -----------------------------------------------------

project = "AcouPipe"
copyright = "Adam Kujawski, Art Pelling, Simon Jekosch, Ennes Sarradj"
author = "Adam Kujawski, Art Pelling, Simon Jekosch, Ennes Sarradj"

# The full version, including alpha/beta/rc tags
release = f"{ap.__version__}"

# -- General configuration ---------------------------------------------------

_SKIP = {"sphinx_gallery.gen_gallery"}  # gallery: re-enable when PR #104 lands
extensions = [
    *(e for e in PACKAGE_FRAME_EXTENSIONS if e not in _SKIP),
    "autoapi.extension",
    "nbsphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx_design",  # tab-set and other design elements
    "traits.util.trait_documenter",
    #"numpydoc", #conda install -c anaconda numpydoc
    "nbsphinx", # allows to include jupyter notebooks into rst documentation
    "sphinxcontrib.bibtex", # to cite papers if necessary
    "sphinx_gallery.gen_gallery", #extension that builds an gallery of examples from Python scripts
]

# auto api configuration
autoapi_type = "python"
autoapi_dirs = [src_dir / "acoupipe"]
autoapi_add_toctree_entry = False  # no seperate index.rst file created by autoapi
autoapi_options = ["show-inheritance"]
autoapi_skip_classes = ["DatasetSyntheticISM", "DatasetSyntheticISMConfig",
    "sample_rms", "sample_mic_noise_variance", "signal_seed", "DatasetSyntheticFeatureCollectionBuilder",
    "ActorHandler", "SamplerActor", "log_execution_time", "bytes_feature", "ConfigBase"]
autoapi_skip_modules = ["acoupipe.datasets.ir"]
autoapi_python_class_content = "both"
# the bibfle
bibtex_bibfiles = ["bib/refs.bib"]

# -- Sphinx-Gallery configuration --------------------------------------------

sphinx_gallery_conf = {
    # Folder(s) containing example scripts
    "examples_dirs": "../../examples",
    # Where the generated .rst + images are written
    "gallery_dirs": "auto_examples",
    # files not matching this pattern will be ignored, and not shown in the gallery
    "filename_pattern": '/example_',
    # Thumbnail size and fallback image matching acoular
    "thumbnail_size": (250, 250),
    'default_thumb_file': str(Path(__file__).parent / '_static' / 'no_image.png'),
    # Order the subsections and files within each subsection in the gallery
    "subsection_order": ExplicitOrder([
        "../../examples/introductory_examples"
        "../../examples/pipelines"
        "../../examples/deployment",
    ]),
    "within_subsection_order": "FileNameSortKey",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

_nav_ctx = build_html_context()
for _link in _nav_ctx["acoular_nav_links"]: # we can remove this workaround once we hardcode the full URL in acoular_sphinx
    if _link["url"].startswith("/"):
        _link["url"] = f"https://www.acoular.org{_link['url']}"

html_context = {
    **_nav_ctx,
    **build_github_context(
        github_user="adku1173",
        github_repo="acoupipe",
        doc_path="docs/source",
        github_version="master",
    ),
}
html_theme_options = configure_package_theme_options(
    package_name="AcouPipe",
    github_url="https://github.com/adku1173/acoupipe",
    pypi_project="acoupipe",
    use_edit_page_button=True,
)
html_sidebars = {"**": ["sidebar-nav-bs.html"]}
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
