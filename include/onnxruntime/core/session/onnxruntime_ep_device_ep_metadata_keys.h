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
// Graph_GetModelMetadata(), e.g., to find out whether the source model may be needed to create a session.
//
// \since Version 1.32.
static const char* const kOrtModelMetadata_WeightlessMode = "weightless_mode";

// Key for the external dependencies of an EPContext model, stored in model metadata.
// Written by ORT when generating an EPContext model. The value lists, separated by ";", the external data that the
// model needs at runtime, resolved from the session and compile options and from how each EP obtains its initializer
// data:
//  - "source_model": the source (original) ONNX model. An EP keeps references to initializers stored inside it,
//    e.g., in weightless mode OrtWeightlessSupport_ALL when the EP loads the initializer data itself
//    (drop_constant_initializers = true). See "ep.context_source_model_path".
//  - "external_weights": external data files with initializer data. Either initializers of the model are stored in
//    external files (e.g., via "ep.context_model_external_initializers_file_name" or
//    ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc), or an EP keeps references to the source
//    model's external data files.
//  - "ep_context_binary": EP context binaries stored outside the model, i.e., EPContext nodes with embed_mode = 0
//    (e.g., with "ep.context_embed_mode" set to "0"). Their "ep_cache_context" attribute names the binary, which the
//    EP may also have provided through an EPContext data write callback instead of a file.
// Values are listed in this order, e.g., "source_model;external_weights;ep_context_binary". The value is empty if the
// model has none of these dependencies.
//
// ORT records the dependencies but does not validate them: missing data is reported when a session is created from
// the model. Applications and tools can read the value via SessionGetModelMetadata() and
// ModelMetadataLookupCustomMetadataMap(), or from the model's metadata_props, e.g., to check what to package with the
// model. Future versions may add values, so readers should expect and ignore values they don't recognize.
//
// \since Version 1.32.
static const char* const kOrtModelMetadata_ExternalDependencies = "external_dependencies";

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
// Values added later, such as "all_or_external_only", are not valid for this key: applications written for earlier
// versions would not recognize them. ORT treats such a value as not recognized. Report them in
// "weightless_supported_modes" instead.
//
// Since version 1.32, "weightless_supported_modes" (kOrtEpDevice_EpMetadataKey_WeightlessSupportedModes) describes
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
// For example, an EP that reported "weightless_support" = "all" before version 1.32 and now also supports
// "external_only" keeps reporting "weightless_support" = "all" and adds
// "weightless_supported_modes" = "all_or_external_only".
//
// Before creating the EP, ORT checks that the mode selected with "ep.enable_weightless_mode" is supported according to
// this entry, or to "weightless_support" if this entry is not present. ORT also checks that the value matches the
// one returned by OrtEp::GetWeightlessSupport() and logs a warning if it does not.
//
// \since Version 1.32.
static const char* const kOrtEpDevice_EpMetadataKey_WeightlessSupportedModes = "weightless_supported_modes";
