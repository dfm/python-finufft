#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import finufft  # NOQA

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "finufft"
author = "Dan Foreman-Mackey, Alex Barnett, and contributors"
copyright = "2017, " + author

version = finufft.__version__
release = finufft.__version__

exclude_patterns = ["_build"]
pygments_style = "sphinx"

# Readthedocs.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_context = dict(
    display_github=True,
    github_user="dfm",
    github_repo="python-finufft",
    github_version="master",
    conf_py_path="/docs/",
)
html_static_path = ["_static"]
html_show_sourcelink = False
