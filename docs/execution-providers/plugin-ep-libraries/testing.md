---
title: Testing Guidance
description: Testing Guidance
grand_parent: Execution Providers
parent: Plugin Execution Provider Libraries
nav_order: 2
---

# Plugin Execution Provider Library Testing Guidance
{: .no_toc }

A plugin EP is responsible for ensuring that its implementation behaves correctly. This includes interacting with ONNX Runtime in the expected way as documented by the plugin EP API. It also includes the operator-level behavior as specified by the operator specification, e.g., from the ONNX standard.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## EP unit testing
Plugin EP implementations are expected to have their own unit tests.

### Operator-level testing utility provided by ONNX Runtime
ONNX Runtime has existing unit tests that validate an EP's op implementation. These tests are located in the unit test program `onnxruntime_provider_test`. This program supports usage with a dynamically specified plugin EP.

In particular, unit tests utilizing the `onnxruntime::test::OpTester` or `onnxruntime::test::ModelTester` classes can also be run with a plugin EP.

Plugin EP implementers may use this test program to help test their plugin EP if desired. The rest of this section explains how to do this.

#### Building
Build the onnxruntime shared library and the `onnxruntime_provider_test` target from source.
```
cd <onnxruntime repo>
# Note: On Windows, use build.bat instead of build.sh
./build.sh --build_shared_lib --update --build --parallel --target onnxruntime_provider_test
```

The onnxruntime shared library and `onnxruntime_provider_test` will be available in the binary output directory.

#### Running
`onnxruntime_provider_test` supports the standard GoogleTest arguments. E.g., `--gtest_filter` can be used to run particular tests of interest.

Importantly, it supports configuration of a dynamically specified plugin EP through the environment variable `ORT_UNIT_TEST_MAIN_DYNAMIC_PLUGIN_EP_CONFIG_JSON`. The configuration value should be specified as a JSON string.

Here is an example value for `ORT_UNIT_TEST_MAIN_DYNAMIC_PLUGIN_EP_CONFIG_JSON`:
```json
{
  "ep_library_registration_name": "example_plugin_ep",
  "ep_library_path": "/path/to/libexample_plugin_ep.so",
  "selected_ep_name": "example_plugin_ep",
  "default_ep_options": { "ep_option_key": "ep_option_value" }
}
```

`ep_library_registration_name` and `ep_library_path` are the same as the `registration_name` and `path` parameters passed in to [`OrtApi::RegisterExecutionProviderLibrary()`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a7c8ea74a2ee54d03052f3d7cd1e1335d), respectively.

`selected_ep_name` should be set to the plugin EP's name. All available `OrtEpDevice`s (returned by [`OrtApi::GetEpDevices()`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a52107386ff1be870f55a0140e6add8dd)) matching that EP name will be used.

As an alternative to `selected_ep_name`, `selected_ep_device_indices` may be set to a list of integers representing the indices into the available `OrtEpDevice`s list. This requires knowing what `OrtEpDevice`s are available.
The available `OrtEpDevices` can be obtained with [`OrtApi::GetEpDevices()`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a52107386ff1be870f55a0140e6add8dd).
The `onnxruntime_perf_test` tool also provides the [`--list_ep_devices`](https://github.com/microsoft/onnxruntime/blob/f83d4d06e4283d53a10c54ce84da3455cfb4e21d/onnxruntime/test/perftest/command_args_parser.cc#L195) option, which may be used in conjunction with the [`--plugin_ep_libs`](https://github.com/microsoft/onnxruntime/blob/f83d4d06e4283d53a10c54ce84da3455cfb4e21d/onnxruntime/test/perftest/command_args_parser.cc#L186-L188) option to display them.

Optionally, `default_ep_options` may be set to specify EP-specific options as string key value pairs.

## EP integration testing and model testing
There are a number of APIs that a plugin EP will implement to interact with ONNX Runtime. Although conformance tests at the EP API level were considered, they were not deemed to be that useful yet. Currently, it is expected that the integration with ONNX Runtime can be meaningfully tested using high level tests that run an entire model.

Plugin EPs may vary significantly from one to another in terms of capability, whether it is optional feature support or operator support. Therefore, it is expected that plugin EPs will have a set of models that are most relevant to them and that these models can be used for testing.
