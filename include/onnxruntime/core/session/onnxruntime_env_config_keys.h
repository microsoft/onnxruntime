// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// This file contains well-known keys for OrtEnv configuration entries, which may be used to configure EPs or
// other global settings.
// Refer to OrtEnvCreationOptions::config_entries and OrtApi::CreateEnvWithOptions.
// This file does NOT specify all available keys as EPs may accept custom entries with the prefix "ep.<ep_name>.".

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
