# ONNX Runtime Plugin Execution Provider Packaging Guidance

## Overview

This document aims to provide guidance for ONNX Runtime (ORT) plugin Execution Provider (EP) implementers to consider with regards to packaging for a plugin EP.

## General Guidance

### Usage

Note: Here, we will refer to the ORT C API functions. Their equivalents should exist for other language bindings that support plugin EP usage.

Users are expected to call the ORT API `RegisterExecutionProviderLibrary()` to register the plugin EP library. Then, they may either choose to use the auto EP selection mechanism or manually call ORT API `SessionOptionsAppendExecutionProvider_V2()` to explicitly use the plugin EP.

### Structure

#### Contents

A plugin EP package should contain the plugin EP shared library file and any other files that need to be distributed with it.

A plugin EP package should NOT contain the ORT shared library or other core ORT libraries (e.g., onnxruntime.dll or libonnxruntime.so). Users should obtain the ORT library separately, most likely via installing the separate ONNX Runtime package.

TODO: Should a plugin EP package have a dependency on the ORT package or be independent?

#### Additional Information to Provide

There should be a way to get the package's plugin EP library path. The user will need the plugin EP library path to call the ORT API `RegisterExecutionProviderLibrary()`. For example, the package may provide a helper function that returns the path to the plugin EP library.

There should be a way to get the package's plugin EP name. The user may require the plugin EP name to select the appropriate `OrtEpDevice` instances to provide to the ORT API `SessionOptionsAppendExecutionProvider_V2()`. For example, the plugin EP name may be well-documented or made available with a helper function provided by the package.

#### Naming

The name of the package should indicate that the package contains a plugin EP and be distinguishable from other ORT packages. For example, this may be done by using a special prefix or suffix.

## Package-specific Guidance

### PyPI

#### Naming

The prefix "onnxruntime-ep" can be used to identify a plugin EP.

The suggested package naming convention is: "onnxruntime-ep-\<EP name\>". For example, "onnxruntime-ep-contoso-ai".

#### TODO Other PyPI stuff

### NuGet
TODO

### Maven
TODO

### Binary Archives
TODO

### TODO other package types?
