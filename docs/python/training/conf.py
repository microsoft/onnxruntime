# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
import onnxruntime
# -- Project information -----------------------------------------------------

project = 'ORTModule'
copyright = '2018-2021, Microsoft'
author = 'Microsoft'
version = onnxruntime.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "alabaster",
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.graphviz',
    'sphinx.ext.napoleon',
    "pyquickhelper.sphinxext.sphinx_runpython_extension",
]

templates_path = ['_templates']
exclude_patterns = []
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_logo = "ONNX_Runtime_icon.png"
html_static_path = ['_static']
graphviz_output_format = "svg"

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
     'examples_dirs': 'examples',
     'gallery_dirs': 'auto_examples',
}
