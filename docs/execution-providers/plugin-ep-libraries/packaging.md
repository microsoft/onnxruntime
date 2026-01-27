---
title: Packaging Guidance
description: Packaging Guidance
grand_parent: Execution Providers
parent: Plugin Execution Provider Libraries
nav_order: 3
---

# Plugin Execution Provider Library Packaging Guidance
{: .no_toc }

This page provides guidance for ONNX Runtime plugin EP implementers to consider with regards to packaging for a plugin EP.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## General Guidance

### Usage
Note: Generally, when referring to the ORT API, we will refer to the C API functions. Equivalents should exist for other language bindings that support plugin EP usage.

#### Manual EP Library Registration
Users are expected to call [`OrtApi::RegisterExecutionProviderLibrary()`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a7c8ea74a2ee54d03052f3d7cd1e1335d) to register the plugin EP library. Then, they may either choose to use the auto EP selection mechanism or manually call [`OrtApi::SessionOptionsAppendExecutionProvider_V2()`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a285a5da8c9a63eff55dc48e4cf3b56f6) to explicitly use the plugin EP.

### Structure

#### Contents
A plugin EP package should contain the plugin EP shared library file and any other files that need to be distributed with it.

A plugin EP package should NOT contain the ORT shared library or other core ORT libraries (e.g., onnxruntime.dll or libonnxruntime.so). Users should obtain the ORT library separately, most likely via installing the separate ONNX Runtime package.

A plugin EP package should have no need to depend on the separate ONNX Runtime package, so it is NOT recommended to do so.

#### Additional Information to Provide

##### Library Path
There should be a way to get the package's plugin EP library path. The user will need the plugin EP library path to call `OrtApi::RegisterExecutionProviderLibrary()`.

For example, the package may provide a helper function that returns the path to the plugin EP library. The recommended name for this helper function is "get library path".

##### EP name(s)
There should be a way to get the plugin EP name(s) provided by the package. The user may require the plugin EP name(s) to select the appropriate `OrtEpDevice` instances to provide to `OrtApi::SessionOptionsAppendExecutionProvider_V2()`.

For example, the plugin EP name(s) may be well-documented or made available with a helper function provided by the package. The recommended name for a helper function returning all EP names is "get EP names". Additionally, if there is only one EP name, a helper function returning the single value named "get EP name" may be provided for convenience.

#### Package Naming
The name of the package should indicate that the package contains a plugin EP and be distinguishable from other ORT packages.

For example, this may be done by using a special prefix or suffix.

## Package-specific Guidance

### PyPI

#### Package Naming
The prefix "onnxruntime-ep" can be used to identify a plugin EP.

The suggested package naming convention is "onnxruntime-ep-\<EP identifier\>".

For example, "onnxruntime-ep-contoso-ai".

#### Helper Functions
The package should provide helper function `get_library_path()` to get the EP library path.

The package should provide helper function `get_ep_names()` to get the EP name(s).

The package may provide helper function `get_ep_name()` to get the single EP name if there is just one.

#### Example
Refer to the [example Python package setup](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/plugin_execution_providers/basic/python) and its [example usage](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/plugin_execution_providers/basic/python/example_usage/example_usage.py).


### NuGet

#### Package Naming
NuGet packages may use a reserved ID prefix.

The suggested package naming convention is "\<Vendor prefix\>.ML.OnnxRuntime.EP.\<EP identifier\>".

For example, "Contoso.ML.OnnxRuntime.EP.ContosoAI".

#### Helper Functions
The package should provide helper function `GetLibraryPath()` to get the EP library path.

The package should provide helper function `GetEpNames()` to get the EP name(s).

The package may provide helper function `GetEpName()` to get the single EP name if there is just one.

#### Example
Refer to the [example NuGet package setup](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/plugin_execution_providers/basic/csharp) and its [example usage](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/plugin_execution_providers/basic/csharp/SampleApp/Program.cs).
