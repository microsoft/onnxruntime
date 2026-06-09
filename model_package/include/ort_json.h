// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file ort_json.h
/// \brief Minimal opaque-handle JSON DOM API exposed by the model_package library.
///
/// Consumers (ORT's CreateSession, GenAI, publisher tools) can parse, navigate,
/// build, mutate, and serialize JSON values without bringing their own JSON
/// dependency. See §11 of model_package_redesign.md for the full design.
///
/// Errors are reported as `ModelPackageStatus*` (the same type used by
/// `ModelPackage_*`). A nullptr return indicates success.
///
/// Lifetime rules:
/// - Values returned by `*New*`, `*Parse*`, and `*ParseFile*` are root handles
///   that the caller MUST `OrtJson_Release()`.
/// - Values returned by navigation accessors (`GetKey`, `ObjectValueAt`,
///   `ArrayAt`) are owned by the parent tree. The caller MUST NOT release them.
/// - On a successful `ObjectSet`/`ArrayAppend`, ownership of the supplied
///   value transfers to the container; the caller MUST NOT release the
///   supplied value (and the pointer becomes invalid).
/// - `const char*` returned by `AsString`, `ObjectKeyAt`, and `Serialize` is
///   owned by the corresponding `OrtJsonValue` and remains valid until either
///   the root is released or a Set/Remove/Append mutates a containing
///   object/array.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "model_package_api.h"  // for MODEL_PACKAGE_API, ModelPackageStatus, ModelPackageErrorCode

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque JSON value handle.
typedef struct OrtJsonValue OrtJsonValue;

/// JSON value type.
typedef enum OrtJsonType {
  ORT_JSON_NULL = 0,
  ORT_JSON_BOOL = 1,
  ORT_JSON_INT = 2,
  ORT_JSON_DOUBLE = 3,
  ORT_JSON_STRING = 4,
  ORT_JSON_ARRAY = 5,
  ORT_JSON_OBJECT = 6
} OrtJsonType;

// ─────────────────────────────────────────────────────────────────────────────
// Parse / serialize / release
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a UTF-8 JSON document from a memory buffer.
/// \param text  Pointer to the start of the buffer. May be non-null-terminated.
/// \param len   Length of the buffer in bytes.
/// \param out   Receives the parsed root on success. Caller releases.
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_Parse(const char* text, size_t len, OrtJsonValue** out);

/// Parse a UTF-8 JSON document from a file on disk.
/// \param path  Null-terminated UTF-8 path.
/// \param out   Receives the parsed root on success. Caller releases.
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_ParseFile(const char* path, OrtJsonValue** out);

/// Serialize a value to a JSON string.
/// \param v          Value to serialize. Must not be null.
/// \param pretty     If true, emit indented multi-line JSON. If false, compact.
/// \param out_text   Receives a pointer to the serialized string. Owned by `v`;
///                   valid until the next mutation of `v` or any of its
///                   descendants, or until `v`'s root is released.
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_Serialize(const OrtJsonValue* v, bool pretty, const char** out_text);

/// Release a root handle. No-op on nullptr. Must NOT be called on values
/// obtained via navigation (`GetKey`, `ObjectValueAt`, `ArrayAt`) or on a
/// value whose ownership has been transferred via `ObjectSet`/`ArrayAppend`.
MODEL_PACKAGE_API void OrtJson_Release(OrtJsonValue* v);

// ─────────────────────────────────────────────────────────────────────────────
// Inspection
// ─────────────────────────────────────────────────────────────────────────────

/// Return the type of `v`. Returns ORT_JSON_NULL for a nullptr input.
MODEL_PACKAGE_API OrtJsonType OrtJson_TypeOf(const OrtJsonValue* v);

/// True iff `obj` is an object that contains `key`.
MODEL_PACKAGE_API bool OrtJson_HasKey(const OrtJsonValue* obj, const char* key);

/// Look up `key` in `obj`. Returns NULL if `obj` is not an object or the key
/// is missing. Result is owned by `obj` (its root, transitively).
MODEL_PACKAGE_API const OrtJsonValue* OrtJson_GetKey(const OrtJsonValue* obj, const char* key);

/// Number of key/value pairs in `obj`. Returns 0 if `obj` is not an object.
MODEL_PACKAGE_API size_t OrtJson_ObjectSize(const OrtJsonValue* obj);

/// Return the key at position `idx` in declaration order. Returns NULL if
/// `obj` is not an object or `idx` is out of range. Owned by `obj`.
MODEL_PACKAGE_API const char* OrtJson_ObjectKeyAt(const OrtJsonValue* obj, size_t idx);

/// Return the value at position `idx` in declaration order. Returns NULL if
/// `obj` is not an object or `idx` is out of range. Owned by `obj`.
MODEL_PACKAGE_API const OrtJsonValue* OrtJson_ObjectValueAt(const OrtJsonValue* obj, size_t idx);

/// Number of elements in `arr`. Returns 0 if `arr` is not an array.
MODEL_PACKAGE_API size_t OrtJson_ArraySize(const OrtJsonValue* arr);

/// Return the element at `idx`. Returns NULL if `arr` is not an array or
/// `idx` is out of range. Owned by `arr`.
MODEL_PACKAGE_API const OrtJsonValue* OrtJson_ArrayAt(const OrtJsonValue* arr, size_t idx);

// ─────────────────────────────────────────────────────────────────────────────
// Typed extraction. Return ERR_SCHEMA if the value is the wrong JSON type.
// ─────────────────────────────────────────────────────────────────────────────

MODEL_PACKAGE_API ModelPackageStatus* OrtJson_AsBool(const OrtJsonValue* v, bool* out);

/// Returns ERR_SCHEMA if the value was parsed/built as a non-integer double
/// (e.g. 3.14), or if it would not fit in int64_t.
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_AsInt(const OrtJsonValue* v, int64_t* out);

/// Accepts both integer and floating-point JSON numbers.
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_AsDouble(const OrtJsonValue* v, double* out);

/// Returns a pointer to a NUL-terminated UTF-8 string. Owned by `v`; valid
/// until mutation of `v` or its containing structure, or release of the root.
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_AsString(const OrtJsonValue* v, const char** out);

// ─────────────────────────────────────────────────────────────────────────────
// Construction. Each returns a fresh root handle (nullptr on OOM).
// ─────────────────────────────────────────────────────────────────────────────

MODEL_PACKAGE_API OrtJsonValue* OrtJson_NewNull(void);
MODEL_PACKAGE_API OrtJsonValue* OrtJson_NewBool(bool b);
MODEL_PACKAGE_API OrtJsonValue* OrtJson_NewInt(int64_t i);
MODEL_PACKAGE_API OrtJsonValue* OrtJson_NewDouble(double d);

/// \param s  Null-terminated UTF-8 string. The contents are copied into the value.
MODEL_PACKAGE_API OrtJsonValue* OrtJson_NewString(const char* s);

MODEL_PACKAGE_API OrtJsonValue* OrtJson_NewArray(void);
MODEL_PACKAGE_API OrtJsonValue* OrtJson_NewObject(void);

// ─────────────────────────────────────────────────────────────────────────────
// Mutation. Ownership of the supplied value transfers to the container on
// success; callers MUST NOT Release a successfully appended/set value.
// On failure, ownership remains with the caller.
// ─────────────────────────────────────────────────────────────────────────────

MODEL_PACKAGE_API ModelPackageStatus* OrtJson_ArrayAppend(OrtJsonValue* arr, OrtJsonValue* item);
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_ObjectSet(OrtJsonValue* obj, const char* key, OrtJsonValue* value);
MODEL_PACKAGE_API ModelPackageStatus* OrtJson_ObjectRemove(OrtJsonValue* obj, const char* key);

#ifdef __cplusplus
}  // extern "C"
#endif
