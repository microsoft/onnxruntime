---
title: Usage
description: Usage
grand_parent: Execution Providers
parent: Plugin Execution Provider Libraries
nav_order: 1
---

# Using a Plugin Execution Provider Library
{: .no_toc }

This page provides a reference on how to use a plugin EP library with the ONNX Runtime API.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Plugin EP library registration
The sample application code below uses the following API functions to register and unregister a plugin EP library.
 - [RegisterExecutionProviderLibrary](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a7c8ea74a2ee54d03052f3d7cd1e1335d)
 - [UnregisterExecutionProviderLibrary](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#acd4d148e149af2f2304a45b65891543f)

```cpp
const char* lib_registration_name = "ep_lib_name";
Ort::Env env;

// Register plugin EP library with ONNX Runtime.
env.RegisterExecutionProviderLibrary(
  lib_registration_name,   // Registration name can be anything the application chooses.
  ORT_TSTR("ep_path.dll")  // Path to the plugin EP library.
);

{
  Ort::Session session(env, /*...*/);
  // Run a model ...
}

// Unregister the library using the application-specified registration name.
// Must only unregister a library after all sessions that use the library have been released.
env.UnregisterExecutionProviderLibrary(lib_registration_name);
```

As shown in the following sequence diagram, registering a plugin EP library causes ONNX Runtime to load the library and
call the library's `CreateEpFactories()` function. During the call to `CreateEpFactories()`, ONNX Runtime determines the subset
of hardware devices supported by each factory by calling `OrtEpFactory::GetSupportedDevices()` with all hardware devices that
ONNX Runtime discovered during initialization.

The factory returns `OrtEpDevice` instances from `OrtEpFactory::GetSupportedDevices()`.
Each `OrtEpDevice` instance pairs a factory with a hardware device that the factory supports.
For example, if a single factory instance supports both CPU and NPU, then the call to `OrtEpFactory::GetSupportedDevices()` returns two `OrtEpDevice` instances:
  - ep_device_0: (factory_0, CPU)
  - ep_device_1: (factory_0, NPU)

<br/>
<p align="center"><img width="100%" src="../../../images/plugin_ep_sd_lib_reg.png" alt="Sequence diagram showing registration and unregistration of a plugin EP library"/></p>

## Session creation with explicit OrtEpDevice(s)
The application code below uses the API function [SessionOptionsAppendExecutionProvider_V2](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a285a5da8c9a63eff55dc48e4cf3b56f6) to add an EP from a library to an ONNX Runtime session.

The application first calls [GetEpDevices](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a52107386ff1be870f55a0140e6add8dd) to get a list of `OrtEpDevices`
available to the application. Each `OrtEpDevice` represents a hardware device supported by an `OrtEpFactory`.
The `SessionOptionsAppendExecutionProvider_V2` function takes an array of `OrtEpDevice` instances as input, where all `OrtEpDevice` instances refer to the same `OrtEpFactory`.

```cpp
Ort::Env env;
env.RegisterExecutionProviderLibrary(/*...*/);

{
  std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

  // Find the Ort::EpDevice for "my_ep".
  std::array<Ort::ConstEpDevice, 1> selected_ep_devices = { nullptr };
  for (Ort::ConstEpDevice ep_device : ep_devices) {
    if (std::strcmp(ep_device.GetName(), "my_ep") == 0) {
      selected_ep_devices[0] = ep_device;
      break;
    }
  }

  if (selected_ep_devices[0] == nullptr) {
    // Did not find EP. Report application error ...
  }

  Ort::KeyValuePairs ep_options(/*...*/);  // Optional EP options.
  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_V2(env, selected_ep_devices, ep_options);

  Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);

  // Run model ...
}

env.UnregisterExecutionProviderLibrary(/*...*/);
```

As shown in the following sequence diagram, ONNX Runtime calls `OrtEpFactory::CreateEp()` during session creation in order to create an instance of the plugin EP.

<br/>
<p align="center"><img width="100%" src="../../../images/plugin_ep_sd_appendv2.png" alt="Sequence diagram showing session creation with explicit ep devices"/></p>

## Session creation with automatic EP selection
The application code below uses the API function [SessionOptionsSetEpSelectionPolicy](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a2ae116df2c6293e4094a6742a6c46f7e) to have ONNX Runtime automatically select an EP based on the user's policy (e.g., PREFER_NPU).
If the plugin EP library registered with ONNX Runtime has a factory that supports NPU, then ONNX Runtime may select an EP from that factory to run the model.

```cpp
Ort::Env env;
env.RegisterExecutionProviderLibrary(/*...*/);

{
  Ort::SessionOptions session_options;
  session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy::PREFER_NPU);

  Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);

  // Run model ...
}

env.UnregisterExecutionProviderLibrary(/*...*/);
```

<br/>
<p align="center"><img width="100%" src="../../../images/plugin_ep_sd_autoep.png" alt="Sequence diagram showing session creation with automatic EP selection"/></p>

## API reference
The following table lists the API functions related to plugin EP library registration and using a plugin EP with a session.

<table>
<tr>
<th>
Function
</th>
<th>
Description
</th>
</tr>

<tr>
<td>
<a href="https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a7c8ea74a2ee54d03052f3d7cd1e1335d">RegisterExecutionProviderLibrary</a>
</td>
<td>
Register an EP library with ORT. The library must export the <code>CreateEpFactories</code> and <code>ReleaseEpFactory</code> functions.
</td>
</tr>

<tr>
<td>
<a href="https://onnxruntime.ai/docs/api/c/struct_ort_api.html#acd4d148e149af2f2304a45b65891543f">UnregisterExecutionProviderLibrary</a>
</td>
<td>
Unregister an EP library with ORT. Caller <b>MUST</b> ensure there are no <code>OrtSession</code> instances using the EPs created by the library before calling this function.
</td>
</tr>

<tr>
<td>
<a href="https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a52107386ff1be870f55a0140e6add8dd">GetEpDevices</a>
</td>
<td>
Get the list of available OrtEpDevice instances.<br/><br/>
Each <code>OrtEpDevice</code> instance contains details of the execution provider and the device it will use.
</td>
</tr>

<tr>
<td>
<a href="https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a285a5da8c9a63eff55dc48e4cf3b56f6">SessionOptionsAppendExecutionProvider_V2</a>
</td>
<td>
Append the execution provider that is responsible for the provided <code>OrtEpDevice</code> instances to the session options.
</td>
</tr>

<tr>
<td>
<a href="https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a2ae116df2c6293e4094a6742a6c46f7e">SessionOptionsSetEpSelectionPolicy</a>
</td>
<td>
Set the execution provider selection policy for the session.<br/><br/>
Allows users to specify a device selection policy for automatic EP selection. If custom selection is required please use
<code>SessionOptionsSetEpSelectionPolicyDelegate</code> instead.
</td>
</tr>

<tr>
<td>
<a href="https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a29c026bc7aa6672f93b7f9e31fd3e4a7">SessionOptionsSetEpSelectionPolicyDelegate</a>
</td>
<td>
Set the execution provider selection policy delegate for the session.<br/><br/>
Allows users to provide a custom device selection policy for automatic EP selection.
</td>
</tr>

</table>
