# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
import onnxruntime
# import recommonmark

# -- Project information -----------------------------------------------------

project = 'ONNX Runtime'
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
    "pyquickhelper.sphinxext.sphinx_runpython_extension",
]

templates_path = ['_templates']

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = ['.rst'] # , '.md']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_logo = "ONNX_Runtime_icon.png"
html_static_path = ['_static']
graphviz_output_format = "svg"

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
    # download examples for the documentation
    this = os.path.abspath(os.path.dirname(__file__))
    dest = os.path.join(this, "model.onnx")
    if not os.path.exists(dest):
        import urllib.request
        url = 'https://raw.githubusercontent.com/onnx/onnx/master/onnx/backend/test/data/node/test_sigmoid/model.onnx'
        urllib.request.urlretrieve(url, dest)
    loc = os.path.split(dest)[-1]
    if not os.path.exists(loc):
        import shutil
        shutil.copy(dest, loc)
    return app

