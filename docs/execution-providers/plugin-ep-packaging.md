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

A plugin EP package should have no need to depend on the separate ONNX Runtime package, so it is NOT recommended to do so.

#### Additional Information to Provide

##### Library Path

There should be a way to get the package's plugin EP library path. The user will need the plugin EP library path to call `OrtApi::RegisterExecutionProviderLibrary()`.

For example, the package may provide a helper function that returns the path to the plugin EP library. The recommended name for this helper function is "get library path".

##### EP name(s)
There should be a way to get the plugin EP name(s) provided by the package. The user may require the plugin EP name(s) to select the appropriate `OrtEpDevice` instances to provide to `OrtApi::SessionOptionsAppendExecutionProvider_V2()`.

For example, the plugin EP name(s) may be well-documented or made available with a helper function provided by the package. The recommended name for this helper function is "get EP name", or "get EP names" if there are multiple names.

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

As mentioned in the general guidance section, the package should provide helper function `get_library_path()` to get the EP library path. The package may provide helper function `get_ep_name()` or `get_ep_names()` to get the EP name(s).

#### Usage example

```python
import onnxruntime as ort
import onnxruntime_ep_contoso_ai as contoso_ep

# Path to the plugin EP library
ep_lib_path = contoso_ep.get_library_path()
# Registration name can be anything the application chooses
ep_registration_name = "contoso_ep_registration"

# Register plugin EP library with ONNX Runtime
ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

# Create ORT session with explicit OrtEpDevice(s)

# Get EP name(s) from the plugin EP library
ep_names = contoso_ep.get_ep_names()
# For this example we'll use the first one
ep_name = ep_names[0]

# Select an OrtEpDevice
# For this example, we'll use any OrtEpDevices matching our EP name
all_ep_devices = ort.get_ep_devices()
selected_ep_devices = [ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_name]

assert len(selected_ep_devices) > 0

sess_options = ort.SessionOptions()

# EP-specific options
ep_options = {}

# Equivalent to the C API's SessionOptionsAppendExecutionProvider_V2 that appends the plugin EP to the session options
sess_options.add_provider_for_devices(selected_ep_devices, ep_options)

assert sess_options.has_providers() == True

# Create ORT session with the plugin EP
model_path = "/path/to/model.onnx"
sess = ort.InferenceSession(model_path, sess_options=sess_options)

# Use `sess`
# ...

del sess

# Unregister the library using the same registration name specified earlier
# Must only unregister a library after all sessions that use the library have been released
ort.unregister_execution_provider_library(ep_registration_name)
```

### NuGet

#### Package Naming

NuGet packages may use a reserved ID prefix.

The suggested package naming convention is "\<Vendor prefix\>.ML.OnnxRuntime.\<EP identifier\>.EP".

For example, "Contoso.ML.OnnxRuntime.ContosoAI.EP".

#### Helper Functions

As mentioned in the general guidance section, the package should provide helper function `GetLibraryPath()` to get the EP library path. The package may provide helper function `GetEpName()` or `GetEpNames()` to get the EP name(s).
