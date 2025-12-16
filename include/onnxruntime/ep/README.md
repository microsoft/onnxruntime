## EP adapter

This folder contains a set of C++ header files. They are used specifically for allowing ONNX Runtime internal kernel-based EPs to use the plugin-style EP API while keep minimal changes to existing code.

### Usage

Make sure to include "ep/pch.h" for all source code in the implementation. Using PCH is recommended.
