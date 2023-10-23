# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0103

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import shutil  # noqa: F401
import sys
from datetime import datetime

# import recommonmark

# -- Project information -----------------------------------------------------

project = "ONNXRuntime-Extensions Python API"
copyright = f"2018-{datetime.now().year}, Microsoft"
author = "Microsoft"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.graphviz",
    "pyquickhelper.sphinxext.sphinx_runpython_extension",
    "sphinxcontrib.googleanalytics",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]

source_parsers = {
    ".md": "recommonmark.parser.CommonMarkParser",
}

source_suffix = [".rst"]  # , '.md']

master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = "default"
autoclass_content = "both"
master_doc = "index"
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_logo = "ONNX_Runtime_icon.png"
html_static_path = ["_static"]
html_theme = "furo"
graphviz_output_format = "svg"

html_context = {
    "default_mode": "auto",  # auto: the documentation theme will follow the system default that you have set (light or dark)
}

html_theme_options = {
    "collapse_navigation": True,
    "external_links": [
        {"name": "onnxruntime", "url": "https://onnxruntime.ai/"},
        {"name": "github", "url": "https://github.com/microsoft/onnxruntime-extensions"},
    ],
    "github_url": "https://github.com/microsoft/onnxruntime-extensions",
    "navbar_center": [],
    "navigation_depth": 5,
    "page_sidebar_items": [],  # default setting is: ["page-toc", "edit-this-page", "sourcelink"],
    "show_nav_level": 0,
    "show_prev_next": True,
    "show_toc_level": 0,
    # needed for sphinx 6.0
    "logo": {
        "text": project,
        "image_light": html_logo,
        "image_dark": html_logo,
        "alt_text": project,
    },
}

# -- Options for Google Analytics -------------------------------------------------

googleanalytics_id = "UA-156955408-1"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "examples",
}