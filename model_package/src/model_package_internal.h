// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_internal.h
/// \brief Internal C++ types for the model package library.

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace model_package {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/// EP compatibility declaration for a variant (opaque to this library).
struct EpCompatibility {
  std::optional<std::string> ep;
  std::optional<std::string> device;
  std::optional<std::string> compatibility_string;
};

/// A single model file within a variant.
struct VariantFile {
  std::string filename;
  std::filesystem::path resolved_path;

  std::optional<std::unordered_map<std::string, std::string>> session_options;
  std::optional<std::unordered_map<std::string, std::string>> provider_options;
  std::optional<std::unordered_map<std::string, std::string>> shared_files;
};

/// A variant of a component.
struct Variant {
  std::string name;
  std::filesystem::path folder_path;
  // Single EP compatibility entry per variant (from metadata.json).
  EpCompatibility ep_compatibility;
  // Single model file entry (from variant.json). Empty when variant.json is absent.
  std::optional<VariantFile> file;
  std::optional<std::string> consumer_metadata_json;
};

/// A component in the model package.
struct Component {
  std::string name;
  std::vector<Variant> variants;
};

/// Top-level model package descriptor.
struct PackageInfo {
  int64_t schema_version{};
  std::filesystem::path root_path;
  std::vector<Component> components;
};

// ─────────────────────────────────────────────────────────────────────────────
// Context implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Internal context holding parsed package data and C API caches.
struct ContextImpl {
  PackageInfo package_info;

  // Caches for C API string access (stable pointers).
  std::vector<std::string> component_names_cache;
  std::unordered_map<std::string, std::vector<std::string>> variant_names_cache;
  std::unordered_map<std::string, std::string> folder_path_strings_cache;

  // Lookup helpers.
  const Component* FindComponent(const char* name) const;
  const Variant* FindVariant(const char* component_name, const char* variant_name) const;
};

}  // namespace model_package
