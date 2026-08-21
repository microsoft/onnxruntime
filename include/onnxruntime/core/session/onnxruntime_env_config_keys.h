// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// This file contains well-known keys for OrtEnv configuration entries, which may be used to configure EPs or
// other global settings.
// Refer to OrtEnvCreationOptions::config_entries and OrtApi::CreateEnvWithOptions.
// This file does NOT specify all available keys. EP-specific environment options use the form
// "ep_factory.<ep_name>.<option>", where <ep_name> is the EP factory's canonical, case-sensitive name and <option>
// is an EP-defined, case-sensitive option name.

// Key for a boolean option that, when enabled, allows EP factories to create virtual OrtHardwareDevice
// instances via OrtEpApi::CreateHardwareDevice().
//
// This config entry is automatically set to "1" by ORT if an application registers an EP library with a registration
// name that ends in the suffix ".virtual". See OrtApi::RegisterExecutionProviderLibrary().
//
// Note: A virtual OrtHardwareDevice does not represent actual hardware on the device, and is identified via the
// metadata entry "is_virtual" with a value of "1".
//
// Allowed values:
//  - "0": Default. Creation of virtual devices is not allowed.
//         This is the assumed default value if this key is not present in the environment's configuration entries.
//  - "1": Creation of virtual devices is allowed.
static const char* const kOrtEnvAllowVirtualDevices = "allow_virtual_devices";

// WebGPU device options. These options configure the default WebGPU context before any session is created.
// Allowed values are "0" (disabled) and "1" (enabled).
// Robustness defaults to enabled in Debug builds and disabled in Release builds.
static const char* const kOrtEnvWebGpuEnableRobustness =
    "ep_factory.WebGpuExecutionProvider.enableRobustness";

// Zero-initialization defaults to enabled.
static const char* const kOrtEnvWebGpuEnableZeroBuffer =
    "ep_factory.WebGpuExecutionProvider.enableZeroBuffer";
