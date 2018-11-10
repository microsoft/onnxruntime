# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
# Check these extensions were installed.
import sphinx_gallery.gen_gallery
# The package should be installed in a virtual environment.
import onnxruntime
# The documentation requires two extensions available at:
# https://github.com/xadupre/sphinx-docfx-yaml
# https://github.com/xadupre/sphinx-docfx-markdown


# -- Project information -----------------------------------------------------

project = 'ONNX Runtime'
copyright = '2018, Microsoft'
author = 'Microsoft'
version = onnxruntime.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
    "docfx_yaml.extension",
    "docfx_markdown",
]

templates_path = ['_templates']

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = ['.rst', '.md']

master_doc = 'intro'
language = "en"
exclude_patterns = []
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

html_static_path = ['_static']
# html_sidebars = {}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
     'examples_dirs': 'examples',
     'gallery_dirs': 'auto_examples',
}

# -- markdown options -----------------------------------------------------------

md_image_dest = "media"
md_link_replace = {
    '#onnxruntimesessionoptionsenable-profiling)': '#class-onnxruntimesessionoptions)',
}

# -- Setup actions -----------------------------------------------------------

def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    return app

