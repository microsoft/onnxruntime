# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0103

"""Configuration file for the Sphinx documentation builder."""

import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "_common"))

# import recommonmark

# -- Project information -----------------------------------------------------

project = "Python API"
copyright = "2018-2024, Microsoft"
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
onnx_doc_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), "operators")
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
        {"name": "github", "url": "https://github.com/microsoft/onnxruntime"},
    ],
    "github_url": "https://github.com/microsoft/onnxruntime",
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
    "gallery_dirs": "auto_examples",
}

# -- markdown options -----------------------------------------------------------

md_image_dest = "media"
md_link_replace = {
    "#onnxruntimesessionoptionsenable-profiling)": "#class-onnxruntimesessionoptions)",
}

# -- Setup actions -----------------------------------------------------------


def setup(app):
    # Build the example Sigmoid model in memory (no network dependency). Previously this
    # urlretrieved onnx/backend/test/data/node/test_sigmoid/model.onnx from GitHub, but that
    # on-disk node-test corpus is being removed upstream (onnx/onnx#7959), which would turn the
    # download into a latent 404. A single-node graph built with onnx.helper is equivalent for
    # the docs and keeps the doc build hermetic. I/O matches the bundled sigmoid example
    # (input "x" -> output "y", float32 [3, 4, 5]).
    this = os.path.abspath(os.path.dirname(__file__))
    dest = os.path.join(this, "model.onnx")
    if not os.path.exists(dest):
        import onnx  # noqa: PLC0415
        from onnx import TensorProto, helper  # noqa: PLC0415

        shape = [3, 4, 5]
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
        node = helper.make_node("Sigmoid", ["x"], ["y"])
        graph = helper.make_graph([node], "sigmoid", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
        model.ir_version = 3
        onnx.save(model, dest)
    loc = os.path.split(dest)[-1]
    if not os.path.exists(loc):
        shutil.copy(dest, loc)
    return app
