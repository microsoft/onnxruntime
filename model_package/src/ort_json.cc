// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file ort_json.cc
/// \brief Implementation of the OrtJson_* opaque-handle DOM API.
///
/// Backed by nlohmann::ordered_json so object key order is preserved across
/// parse and round-trip.
///
/// Internal representation
/// -----------------------
/// `OrtJsonValue` is one of:
///   - A root: owns its underlying ordered_json via `storage`.
///   - A view: borrows a pointer into a parent root's tree (`storage` empty).
///
/// To make navigation idempotent and cheap, every container caches its child
/// views in per-key (objects) or per-index (arrays) maps. Pointers into a
/// container remain valid until the container itself is mutated.
///
/// Mutation invalidation is scoped per the design: a Set/Remove on object X
/// invalidates pointers into X and (transitively) into X's children, but not
/// pointers into unrelated subtrees. We implement that by clearing the view
/// cache of the mutated container; transitive invalidation follows naturally
/// because the cleared children are unique_ptr-owned and their own view caches
/// destruct with them.
///
/// String pointers returned by AsString / ObjectKeyAt / Serialize either point
/// directly into the ordered_json storage (for AsString and ObjectKeyAt, where
/// nlohmann stores strings inline) or into a per-value Serialize cache.

#include "ort_json.h"

#include <new>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <list>
#include <cmath>

#include <nlohmann/json.hpp>

#include "status_impl.h"

using nlohmann::ordered_json;
using model_package::MakeStatus;

// ─────────────────────────────────────────────────────────────────────────────
// OrtJsonValue
// ─────────────────────────────────────────────────────────────────────────────

struct OrtJsonValue {
  // The actual JSON data this value refers to.
  ordered_json* node{nullptr};

  // For roots: the owned storage that `node` points into.
  std::unique_ptr<ordered_json> storage;

  // View cache. Each container caches OrtJsonValue wrappers for the children
  // that have been navigated into, keyed by object key or array index. We use
  // ordered_map / std::map so iterators are stable on insertion.
  std::map<std::string, std::unique_ptr<OrtJsonValue>> obj_children;
  std::map<size_t, std::unique_ptr<OrtJsonValue>> arr_children;

  // Cache of serialized strings returned via OrtJson_Serialize. Stored in a
  // std::list so existing pointers stay valid as new entries are appended.
  std::list<std::string> serialize_cache;

  // Cleared on any mutation of this node. Transitive invalidation is implicit:
  // freeing a child unique_ptr also destroys its descendant view caches.
  void InvalidateChildViews() {
    obj_children.clear();
    arr_children.clear();
    serialize_cache.clear();
  }
};

namespace {

OrtJsonValue* NewRoot(ordered_json j) {
  auto v = new (std::nothrow) OrtJsonValue();
  if (!v) return nullptr;
  v->storage = std::make_unique<ordered_json>(std::move(j));
  v->node = v->storage.get();
  return v;
}

OrtJsonValue* MakeView(OrtJsonValue& parent_owner, ordered_json* node_ptr) {
  auto v = std::make_unique<OrtJsonValue>();
  v->node = node_ptr;
  auto* raw = v.get();
  (void)parent_owner;  // ownership handled by caller via obj_children/arr_children
  return v.release();  // caller transfers into the cache map
}

// Returns true if `obj` is non-null and wraps a JSON object.
bool IsObjectValue(const OrtJsonValue* obj) {
  return obj && obj->node && obj->node->is_object();
}

bool IsArrayValue(const OrtJsonValue* arr) {
  return arr && arr->node && arr->node->is_array();
}

ModelPackageStatus* TypeMismatch(const char* op, const char* expected) {
  std::string msg = "OrtJson: ";
  msg += op;
  msg += " requires a JSON ";
  msg += expected;
  msg += " value.";
  return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA, std::move(msg));
}

ModelPackageStatus* NullArg(const char* name) {
  std::string msg = "OrtJson: '";
  msg += name;
  msg += "' must not be null.";
  return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG, std::move(msg));
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Parse / serialize / release
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

ModelPackageStatus* OrtJson_Parse(const char* text, size_t len, OrtJsonValue** out) {
  if (!text) return NullArg("text");
  if (!out) return NullArg("out");
  *out = nullptr;
  try {
    ordered_json j = ordered_json::parse(text, text + len);
    auto* root = NewRoot(std::move(j));
    if (!root) return MakeStatus(MODEL_PACKAGE_ERR_IO, "OrtJson_Parse: out of memory.");
    *out = root;
    return nullptr;
  } catch (const ordered_json::parse_error& e) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      std::string("OrtJson_Parse: ") + e.what());
  } catch (const std::exception& e) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      std::string("OrtJson_Parse: ") + e.what());
  }
}

ModelPackageStatus* OrtJson_ParseFile(const char* path, OrtJsonValue** out) {
  if (!path) return NullArg("path");
  if (!out) return NullArg("out");
  *out = nullptr;
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      std::string("OrtJson_ParseFile: cannot open '") + path + "'.");
  }
  std::ostringstream buf;
  buf << f.rdbuf();
  std::string text = buf.str();
  return OrtJson_Parse(text.data(), text.size(), out);
}

ModelPackageStatus* OrtJson_Serialize(const OrtJsonValue* v, bool pretty, const char** out_text) {
  if (!v) return NullArg("v");
  if (!out_text) return NullArg("out_text");
  *out_text = nullptr;
  try {
    auto* mut = const_cast<OrtJsonValue*>(v);
    std::string s = v->node->dump(pretty ? 2 : -1);
    mut->serialize_cache.push_back(std::move(s));
    *out_text = mut->serialize_cache.back().c_str();
    return nullptr;
  } catch (const std::exception& e) {
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      std::string("OrtJson_Serialize: ") + e.what());
  }
}

void OrtJson_Release(OrtJsonValue* v) {
  // Roots own their storage; deleting them also clears all view caches.
  // Views (`!storage`) should not be released by the caller per the API
  // contract, but we tolerate it by being a no-op to avoid double-frees:
  // they will be cleaned up when their owning root is released.
  if (!v) return;
  if (!v->storage) return;  // view: not ours to delete
  delete v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inspection
// ─────────────────────────────────────────────────────────────────────────────

OrtJsonType OrtJson_TypeOf(const OrtJsonValue* v) {
  if (!v || !v->node) return ORT_JSON_NULL;
  switch (v->node->type()) {
    case ordered_json::value_t::null:            return ORT_JSON_NULL;
    case ordered_json::value_t::boolean:         return ORT_JSON_BOOL;
    case ordered_json::value_t::number_integer:  return ORT_JSON_INT;
    case ordered_json::value_t::number_unsigned: return ORT_JSON_INT;
    case ordered_json::value_t::number_float:    return ORT_JSON_DOUBLE;
    case ordered_json::value_t::string:          return ORT_JSON_STRING;
    case ordered_json::value_t::array:           return ORT_JSON_ARRAY;
    case ordered_json::value_t::object:          return ORT_JSON_OBJECT;
    default:                                     return ORT_JSON_NULL;
  }
}

bool OrtJson_HasKey(const OrtJsonValue* obj, const char* key) {
  if (!IsObjectValue(obj) || !key) return false;
  return obj->node->contains(key);
}

const OrtJsonValue* OrtJson_GetKey(const OrtJsonValue* obj, const char* key) {
  if (!IsObjectValue(obj) || !key) return nullptr;
  auto it = obj->node->find(key);
  if (it == obj->node->end()) return nullptr;

  auto* mut = const_cast<OrtJsonValue*>(obj);
  std::string k(key);
  auto cached = mut->obj_children.find(k);
  if (cached != mut->obj_children.end()) {
    // The underlying ordered_json node might have moved if the object was
    // mutated, but we clear the cache on mutation, so a hit here is valid.
    return cached->second.get();
  }
  auto view_uptr = std::unique_ptr<OrtJsonValue>(MakeView(*mut, &(*it)));
  if (!view_uptr) return nullptr;
  auto* raw = view_uptr.get();
  mut->obj_children.emplace(std::move(k), std::move(view_uptr));
  return raw;
}

size_t OrtJson_ObjectSize(const OrtJsonValue* obj) {
  if (!IsObjectValue(obj)) return 0;
  return obj->node->size();
}

const char* OrtJson_ObjectKeyAt(const OrtJsonValue* obj, size_t idx) {
  if (!IsObjectValue(obj) || idx >= obj->node->size()) return nullptr;
  auto it = obj->node->begin();
  std::advance(it, static_cast<std::ptrdiff_t>(idx));
  // it.key() returns a reference to the stored key string; lifetime tied to
  // the parent object, invalidated on mutation per the contract.
  return it.key().c_str();
}

const OrtJsonValue* OrtJson_ObjectValueAt(const OrtJsonValue* obj, size_t idx) {
  if (!IsObjectValue(obj) || idx >= obj->node->size()) return nullptr;
  auto it = obj->node->begin();
  std::advance(it, static_cast<std::ptrdiff_t>(idx));
  return OrtJson_GetKey(obj, it.key().c_str());
}

size_t OrtJson_ArraySize(const OrtJsonValue* arr) {
  if (!IsArrayValue(arr)) return 0;
  return arr->node->size();
}

const OrtJsonValue* OrtJson_ArrayAt(const OrtJsonValue* arr, size_t idx) {
  if (!IsArrayValue(arr) || idx >= arr->node->size()) return nullptr;

  auto* mut = const_cast<OrtJsonValue*>(arr);
  auto cached = mut->arr_children.find(idx);
  if (cached != mut->arr_children.end()) {
    return cached->second.get();
  }
  ordered_json* node_ptr = &(*arr->node)[idx];
  auto view_uptr = std::unique_ptr<OrtJsonValue>(MakeView(*mut, node_ptr));
  if (!view_uptr) return nullptr;
  auto* raw = view_uptr.get();
  mut->arr_children.emplace(idx, std::move(view_uptr));
  return raw;
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed extraction
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* OrtJson_AsBool(const OrtJsonValue* v, bool* out) {
  if (!v) return NullArg("v");
  if (!out) return NullArg("out");
  if (!v->node->is_boolean()) return TypeMismatch("OrtJson_AsBool", "boolean");
  *out = v->node->get<bool>();
  return nullptr;
}

ModelPackageStatus* OrtJson_AsInt(const OrtJsonValue* v, int64_t* out) {
  if (!v) return NullArg("v");
  if (!out) return NullArg("out");
  if (v->node->is_number_integer() || v->node->is_number_unsigned()) {
    if (v->node->is_number_unsigned()) {
      uint64_t u = v->node->get<uint64_t>();
      if (u > static_cast<uint64_t>(INT64_MAX)) {
        return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                          "OrtJson_AsInt: value exceeds int64_t range.");
      }
      *out = static_cast<int64_t>(u);
    } else {
      *out = v->node->get<int64_t>();
    }
    return nullptr;
  }
  return TypeMismatch("OrtJson_AsInt", "integer");
}

ModelPackageStatus* OrtJson_AsDouble(const OrtJsonValue* v, double* out) {
  if (!v) return NullArg("v");
  if (!out) return NullArg("out");
  if (!v->node->is_number()) return TypeMismatch("OrtJson_AsDouble", "number");
  *out = v->node->get<double>();
  return nullptr;
}

ModelPackageStatus* OrtJson_AsString(const OrtJsonValue* v, const char** out) {
  if (!v) return NullArg("v");
  if (!out) return NullArg("out");
  if (!v->node->is_string()) return TypeMismatch("OrtJson_AsString", "string");
  // get_ref returns a reference to the stored string; pointer remains valid
  // until the value is mutated or its root is released.
  *out = v->node->get_ref<const std::string&>().c_str();
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

OrtJsonValue* OrtJson_NewNull(void)        { return NewRoot(ordered_json(nullptr)); }
OrtJsonValue* OrtJson_NewBool(bool b)      { return NewRoot(ordered_json(b)); }
OrtJsonValue* OrtJson_NewInt(int64_t i)    { return NewRoot(ordered_json(i)); }
OrtJsonValue* OrtJson_NewDouble(double d)  { return NewRoot(ordered_json(d)); }
OrtJsonValue* OrtJson_NewString(const char* s) {
  if (!s) return NewRoot(ordered_json(std::string()));
  return NewRoot(ordered_json(std::string(s)));
}
OrtJsonValue* OrtJson_NewArray(void)       { return NewRoot(ordered_json::array()); }
OrtJsonValue* OrtJson_NewObject(void)      { return NewRoot(ordered_json::object()); }

// ─────────────────────────────────────────────────────────────────────────────
// Mutation
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* OrtJson_ArrayAppend(OrtJsonValue* arr, OrtJsonValue* item) {
  if (!arr) return NullArg("arr");
  if (!item) return NullArg("item");
  if (!arr->node || !arr->node->is_array()) {
    return TypeMismatch("OrtJson_ArrayAppend", "array");
  }
  if (!item->storage) {
    // Item is a view, not a root: cannot transfer ownership.
    return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                      "OrtJson_ArrayAppend: 'item' must be a root value created by an "
                      "OrtJson_New*/OrtJson_Parse* function, not a view returned by a "
                      "navigation accessor.");
  }
  try {
    arr->node->push_back(std::move(*item->node));
  } catch (const std::exception& e) {
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      std::string("OrtJson_ArrayAppend: ") + e.what());
  }
  arr->InvalidateChildViews();
  // Consume the item.
  delete item;
  return nullptr;
}

ModelPackageStatus* OrtJson_ObjectSet(OrtJsonValue* obj, const char* key, OrtJsonValue* value) {
  if (!obj) return NullArg("obj");
  if (!key) return NullArg("key");
  if (!value) return NullArg("value");
  if (!obj->node || !obj->node->is_object()) {
    return TypeMismatch("OrtJson_ObjectSet", "object");
  }
  if (!value->storage) {
    return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                      "OrtJson_ObjectSet: 'value' must be a root value created by an "
                      "OrtJson_New*/OrtJson_Parse* function, not a view returned by a "
                      "navigation accessor.");
  }
  try {
    (*obj->node)[key] = std::move(*value->node);
  } catch (const std::exception& e) {
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      std::string("OrtJson_ObjectSet: ") + e.what());
  }
  obj->InvalidateChildViews();
  delete value;
  return nullptr;
}

ModelPackageStatus* OrtJson_ObjectRemove(OrtJsonValue* obj, const char* key) {
  if (!obj) return NullArg("obj");
  if (!key) return NullArg("key");
  if (!obj->node || !obj->node->is_object()) {
    return TypeMismatch("OrtJson_ObjectRemove", "object");
  }
  obj->node->erase(key);
  obj->InvalidateChildViews();
  return nullptr;
}

}  // extern "C"
