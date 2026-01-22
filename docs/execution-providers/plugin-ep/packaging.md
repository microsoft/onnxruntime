# ONNX Runtime Plugin Execution Provider Packaging Guidance

## Overview

This document aims to provide guidance for ONNX Runtime (ORT) plugin Execution Provider (EP) implementers to consider with regards to packaging for a plugin EP.

## General Guidance

### Usage

Note: Generally, when referring to the ORT API, we will refer to the C API functions. Equivalents should exist for other language bindings that support plugin EP usage.

#### Manual EP Library Registration

Users are expected to call [`OrtApi::RegisterExecutionProviderLibrary()`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a7c8ea74a2ee54d03052f3d7cd1e1335d) to register the plugin EP library. Then, they may either choose to use the auto EP selection mechanism or manually call [`OrtApi::SessionOptionsAppendExecutionProvider_V2()`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a285a5da8c9a63eff55dc48e4cf3b56f6) to explicitly use the plugin EP.

### Structure

#### Contents

A plugin EP package should contain the plugin EP shared library file and any other files that need to be distributed with it.

A plugin EP package should NOT contain the ORT shared library or other core ORT libraries (e.g., onnxruntime.dll or libonnxruntime.so). Users should obtain the ORT library separately, most likely via installing the separate ONNX Runtime package.

TODO: Should a plugin EP package have a dependency on the ORT package or be independent?

##### Shared Library File Naming

The suggested plugin EP shared library file naming convention is "onnxruntime_ep_\<EP identifier\>" for the base name with the appropriate platform-specific prefixes or suffixes.

For example, "onnxruntime_ep_contoso_ai.dll", "libonnxruntime_ep_contoso_ai.so", or "libonnxruntime_ep_contoso_ai.dylib".

#### Additional Information to Provide

There should be a way to get the package's plugin EP library path. The user will need the plugin EP library path to call `OrtApi::RegisterExecutionProviderLibrary()`. For example, the package may provide a helper function that returns the path to the plugin EP library.

There should be a way to get the package's plugin EP name. The user may require the plugin EP name to select the appropriate `OrtEpDevice` instances to provide to `OrtApi::SessionOptionsAppendExecutionProvider_V2()`. For example, the plugin EP name may be well-documented or made available with a helper function provided by the package.

#### Package Naming

The name of the package should indicate that the package contains a plugin EP and be distinguishable from other ORT packages.

For example, this may be done by using a special prefix or suffix.

## Package-specific Guidance

### PyPI

#### Package Naming

The prefix "onnxruntime-ep" can be used to identify a plugin EP.

The suggested package naming convention is "onnxruntime-ep-\<EP identifier\>".

For example, "onnxruntime-ep-contoso-ai".

#### TODO other PyPI info

### NuGet

#### Package Naming

NuGet packages may use a reserved ID prefix.

The suggested package naming convention is "\<Vendor prefix\>.ML.OnnxRuntime.\<EP identifier\>.EP".

For example, "Contoso.ML.OnnxRuntime.ContosoAI.EP".

#### TODO other NuGet info

### Maven
TODO

### Binary Archives
TODO

### TODO other package types?
