# ONNX Runtime C/C++ docs source files

This directory contains doxygen configuration to generate the C/C++ API docs for ONNX Runtime.

The actual generation is performed by a GitHub actions workflow: [publish-c-apidocs.yml](../../.github/workflows/publish-c-apidocs.yml).

The workflow is currently manually triggered, and generates a PR to the gh-pages branch, for publication on https://onnxruntime.ai.