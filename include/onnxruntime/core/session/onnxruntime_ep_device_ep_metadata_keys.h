// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// This file contains well-known keys for OrtEpDevice and OrtHardwareDevice metadata entries.
// It does NOT specify all available metadata keys.

// Key for the execution provider version string. This should be available for all plugin EPs.
static const char* const kOrtEpDevice_EpMetadataKey_Version = "version";

// Key for the execution provider OS driver version.
// Value should be a 4-part dot-separated version string in the format "a.b.c.d" (e.g., "31.0.101.4502").
// This maps to the Windows DXCore adapter property DXCoreAdapterProperty::DriverVersion
// (https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoreadapterproperty).
// On non-Windows platforms, the EP should provide an equivalent OS-level driver version if available.
static const char* const kOrtEpDevice_EpMetadataKey_OSDriverVersion = "os_driver_version";

// Prefix for execution provider compatibility information stored in model metadata.
// Used when generating EP context models to store compatibility strings for each EP.
// Full key format: "ep_compatibility_info.<EP_TYPE>"
static const char* const kOrtModelMetadata_EpCompatibilityInfoPrefix = "ep_compatibility_info.";

// Key for the weightless mode an EPContext model was compiled with, stored in model metadata.
// Written by ORT when generating an EPContext model with "ep.enable_weightless_mode"
// (kOrtSessionOptionEpEnableWeightlessMode) set to a mode other than OrtWeightlessSupport_NONE.
// The value is the base-10 string representation of the OrtWeightlessSupport value (e.g., "2" for
// OrtWeightlessSupport_ALL). The key is absent if the model was compiled without weightless mode, or with the
// deprecated "ep.enable_weightless" option, which does not select a mode.
//
// Applications can read it via SessionGetModelMetadata()/ModelMetadataLookupCustomMetadataMap(), and EPs via
// Graph_GetModelMetadata(), e.g., to find out whether the source model is needed to create a session.
//
// \since Version 1.31.
static const char* const kOrtModelMetadata_WeightlessMode = "weightless_mode";

// Key for the execution provider library path (for dynamically loaded EPs)
static const char* const kOrtEpDevice_EpMetadataKey_LibraryPath = "library_path";

// Optional metadata key for the execution provider's preferred layout of the Value KV-cache tensors
// (the past_value input and present_value output) of com.microsoft.GroupQueryAttention.
// Possible values:
//  - "BNSH": (batch_size, num_heads, sequence_length, head_size). This is the assumed default value
//            if this metadata key is not present, and matches the operator schema.
//  - "BNHS": (batch_size, num_heads, head_size, sequence_length).
// An EP that reports "BNHS" is expected to fuse the Transpose -> GroupQueryAttention -> Transpose
// sequence that ORT inserts when the application selects that layout.
// The application passes the layout it has chosen to the session via the
// kOrtSessionOptionsGqaValueLayout session option (see onnxruntime_session_options_config_keys.h).
static const char* const kOrtEpDevice_EpMetadataKey_GqaPreferredValueLayout = "gqa_preferred_value_layout";

// Optional metadata key to determine if a OrtHardwareDevice represents a virtual (non-hardware) device.
// Possible values:
//  - "0": OrtHardwareDevice is not virtual (i.e., actual hardware device). This is the assumed default value
//         if this metadata key is not present.
//  - "1": OrtHardwareDevice is virtual.
static const char* const kOrtHardwareDevice_MetadataKey_IsVirtual = "is_virtual";

// Key for the execution provider's weightless mode support on a specific device.
// The app can read it to check device-specific weightless capability
// before calling ModelCompilationOptions_SetWeightlessEnabled() or specifying kOrtSessionOptionEpEnableWeightlessMode.
//
// Possible values:
//  - "none": EP does not support weightless mode on this device. This is the assumed default value
//            if this metadata key is not present.
//  - "external_only": EP supports weightless mode for external initializers only (e.g., older
//                     hardware/driver that must transform internal constants).
//  - "all": EP supports weightless mode for all initializers (internal and external).
//
// Since version 1.31, "weightless_supported_modes" (kOrtEpDevice_EpMetadataKey_WeightlessSupportedModes) describes
// all the modes an EP supports. EPs should keep reporting this key, with the value they reported before, so that
// applications written for earlier versions keep working. Applications should prefer the newer key when present.
//
// \since Version 1.29.
static const char* const kOrtEpDevice_EpMetadataKey_WeightlessSupport = "weightless_support";

// Key for all the weightless modes the execution provider supports on a specific device.
// The app can read it to check the device-specific weightless modes and pick the one
// that best fits its use case (or none at all) before calling ModelCompilationOptions_SetWeightlessMode()
// or setting the "ep.enable_weightless_mode" session option (kOrtSessionOptionEpEnableWeightlessMode).
// If this key is not present, the app falls back to "weightless_support"
// (kOrtEpDevice_EpMetadataKey_WeightlessSupport).
//
// Possible values:
//  - "none": EP does not support weightless mode on this device (OrtWeightlessSupport_NONE).
//  - "external_only": EP supports weightless mode for external initializers only (OrtWeightlessSupport_EXTERNAL_ONLY).
//  - "all": EP supports weightless mode for all initializers (OrtWeightlessSupport_ALL).
//  - "all_or_external_only": EP supports both "external_only" and "all" (OrtWeightlessSupport_ALL_OR_EXTERNAL_ONLY).
//
// Future versions may add values. Applications should expect values they don't recognize and ignore them, i.e.,
// not select a weightless mode based on such a value.
//
// For example, an EP that reported "weightless_support" = "all" before version 1.31 and now also supports
// "external_only" keeps reporting "weightless_support" = "all" and adds
// "weightless_supported_modes" = "all_or_external_only".
//
// Before creating the EP, ORT checks that the mode selected with "ep.enable_weightless_mode" is supported according to
// this entry, or to "weightless_support" if this entry is not present. ORT also checks that the value matches the
// one returned by OrtEp::GetWeightlessSupport() and logs a warning if it does not.
//
// \since Version 1.31.
static const char* const kOrtEpDevice_EpMetadataKey_WeightlessSupportedModes = "weightless_supported_modes";
