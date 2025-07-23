// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/ort_api.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <gsl/gsl>

namespace onnxruntime {

OrtNodeUnit::OrtNodeUnit(const OrtNode& node, const OrtApi& ort_api) : target_node_(node), type_(Type::SingleNode) {
  InitForSingleNode(ort_api);
}

void OrtNodeUnit::InitForSingleNode(const OrtApi& ort_api) {
  OrtArrayOfConstObjects* inputs_array = nullptr;
  OrtArrayOfConstObjects* outputs_array = nullptr;

  ort_api.Node_GetInputs(&target_node_, &inputs_array);
  ort_api.Node_GetOutputs(&target_node_, &outputs_array);

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  ort_api.ArrayOfConstObjects_GetSize(inputs_array, &num_inputs);
  ort_api.ArrayOfConstObjects_GetSize(outputs_array, &num_outputs);

  const void* const* inputs_data = nullptr;
  const void* const* outputs_data = nullptr;
  ort_api.ArrayOfConstObjects_GetData(inputs_array, &inputs_data);
  ort_api.ArrayOfConstObjects_GetData(outputs_array, &outputs_data);

  auto add_io_def = [&](std::vector<OrtNodeUnitIODef>& io_defs, const void* const* data, size_t num_data) {
    for (size_t idx = 0; idx < num_data; ++idx) {
      const OrtValueInfo* io = static_cast<const OrtValueInfo*>(data[idx]);

      // Get name.
      const char* name = nullptr;
      ort_api.GetValueInfoName(io, &name);

      // Get type and shape.
      const OrtTypeInfo* type_info = nullptr;
      ort_api.GetValueInfoTypeInfo(io, &type_info);
      const OrtTensorTypeAndShapeInfo* type_shape = nullptr;
      ort_api.CastTypeInfoToTensorInfo(type_info, &type_shape);

      ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      ort_api.GetTensorElementType(type_shape, &elem_type);

      size_t num_dims = 0;
      ort_api.GetDimensionsCount(type_shape, &num_dims);

      std::vector<int64_t> shape;
      shape.resize(num_dims, 0);
      ort_api.GetDimensions(type_shape, shape.data(), shape.size());

      io_defs.push_back(OrtNodeUnitIODef{name, elem_type, shape});

      // TODO: SegFault if enabled release.
      // ort_api.ReleaseTensorTypeAndShapeInfo(const_cast<OrtTensorTypeAndShapeInfo*>(type_shape));
      // ort_api.ReleaseTypeInfo(const_cast<OrtTypeInfo*>(type_info));
    }
  };

  inputs_.reserve(num_inputs);
  add_io_def(inputs_, inputs_data, num_inputs);

  outputs_.reserve(num_outputs);
  add_io_def(outputs_, outputs_data, num_outputs);

  ort_api.ReleaseArrayOfConstObjects(inputs_array);
  ort_api.ReleaseArrayOfConstObjects(outputs_array);
}

// std::vector<const Node*> Graph__Nodes(const Graph& graph) {
//   return graph.Nodes();
// }

#define NODE_ATTR_ITER_VAL(iter) (iter)->second()

OrtNodeAttrHelper::OrtNodeAttrHelper(const OrtApi& ort_api, const OrtNode& node) : node_(node), ort_api_(ort_api) {}

OrtNodeAttrHelper::OrtNodeAttrHelper(const OrtApi& ort_api, const OrtNodeUnit& node_unit) : node_(node_unit.GetNode()), ort_api_(ort_api) {}

float OrtNodeAttrHelper::Get(const std::string& key, float def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  return rt ? def_val : api_node_attr->attr_proto.f();
}

int32_t OrtNodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  return rt ? def_val : gsl::narrow<int32_t>(api_node_attr->attr_proto.i());
}

uint32_t OrtNodeAttrHelper::Get(const std::string& key, uint32_t def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  return rt ? def_val : gsl::narrow<uint32_t>(api_node_attr->attr_proto.i());
}

int64_t OrtNodeAttrHelper::Get(const std::string& key, int64_t def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  return rt ? def_val : api_node_attr->attr_proto.i();
}

const std::string& OrtNodeAttrHelper::Get(const std::string& key, const std::string& def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return def_val;
  }
  static std::string result = api_node_attr->attr_proto.s();
  return result;
}

std::vector<std::string> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<std::string>& def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return def_val;
  }

  const auto& strings_proto = api_node_attr->attr_proto.strings();
  std::vector<std::string> result;
  result.reserve(strings_proto.size());
  for (int i = 0; i < strings_proto.size(); ++i) {
    result.emplace_back(strings_proto.Get(i));
  }
  return result;
}

std::vector<int32_t> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<int32_t>& def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return def_val;
  }

  const auto& ints_proto = api_node_attr->attr_proto.ints();
  std::vector<int32_t> result;
  result.reserve(ints_proto.size());
  for (int i = 0; i < ints_proto.size(); ++i) {
    result.push_back(gsl::narrow<int32_t>(ints_proto.Get(i)));
  }
  return result;
}

std::vector<uint32_t> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<uint32_t>& def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return def_val;
  }

  const auto& ints_proto = api_node_attr->attr_proto.ints();
  std::vector<uint32_t> result;
  result.reserve(ints_proto.size());
  for (int i = 0; i < ints_proto.size(); ++i) {
    result.push_back(gsl::narrow<uint32_t>(ints_proto.Get(i)));
  }
  return result;
}

std::vector<int64_t> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<int64_t>& def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return def_val;
  }

  const auto& ints_proto = api_node_attr->attr_proto.ints();
  std::vector<int64_t> result;
  result.reserve(ints_proto.size());
  for (int i = 0; i < ints_proto.size(); ++i) {
    result.push_back(ints_proto.Get(i));
  }
  return result;
}

std::vector<float> OrtNodeAttrHelper::Get(const std::string& key, const std::vector<float>& def_val) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return def_val;
  }

  const auto& floats_proto = api_node_attr->attr_proto.floats();
  std::vector<float> result;
  result.reserve(floats_proto.size());
  for (int i = 0; i < floats_proto.size(); ++i) {
    result.push_back(floats_proto.Get(i));
  }
  return result;
}

std::optional<float> OrtNodeAttrHelper::GetFloat(const std::string& key) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return std::nullopt;
  }
  return api_node_attr->attr_proto.f();
}

std::optional<int64_t> OrtNodeAttrHelper::GetInt64(const std::string& key) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return std::nullopt;
  }
  return api_node_attr->attr_proto.i();
}

std::optional<std::vector<float>> OrtNodeAttrHelper::GetFloats(const std::string& key) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return std::nullopt;
  }

  const auto& floats_proto = api_node_attr->attr_proto.floats();
  std::vector<float> result;
  result.reserve(floats_proto.size());
  for (int i = 0; i < floats_proto.size(); ++i) {
    result.push_back(floats_proto.Get(i));
  }
  return result;
}

std::optional<std::vector<int64_t>> OrtNodeAttrHelper::GetInt64s(const std::string& key) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return std::nullopt;
  }

  const auto& ints_proto = api_node_attr->attr_proto.ints();
  std::vector<int64_t> result;
  result.reserve(ints_proto.size());
  for (int i = 0; i < ints_proto.size(); ++i) {
    result.push_back(ints_proto.Get(i));
  }
  return result;
}

std::optional<std::string> OrtNodeAttrHelper::GetString(const std::string& key) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  if (rt) {
    return std::nullopt;
  }
  return api_node_attr->attr_proto.s();
}

bool OrtNodeAttrHelper::HasAttr(const std::string& key) const {
  const OrtOpAttr* api_node_attr = nullptr;
  auto rt = ort_api_.Node_GetAttributeByName(&node_, key.c_str(), &api_node_attr);
  return !rt;  // Return true if attribute exists (rt == 0), false if not found (rt != 0)
}

PathString OrtGetRuntimePath() {
#ifdef _WIN32
  IMAGE_DOS_HEADER __ImageBase;
  wchar_t buffer[MAX_PATH];
  if (!GetModuleFileNameW(reinterpret_cast<HINSTANCE>(&__ImageBase), buffer, _countof(buffer))) {
    return PathString();
  }

  // Remove the filename at the end, but keep the trailing slash
  PathString path(buffer);
  auto slash_index = path.find_last_of(ORT_TSTR('\\'));
  if (slash_index == std::string::npos) {
    // Windows supports forward slashes
    slash_index = path.find_last_of(ORT_TSTR('/'));
    if (slash_index == std::string::npos) {
      return PathString();
    }
  }
  return path.substr(0, slash_index + 1);
#else
  return PathString();
#endif
}

Status OrtLoadDynamicLibrary(const PathString& wlibrary_filename, bool /* global_symbols */, void** handle) {
#ifdef _WIN32
#if WINAPI_FAMILY == WINAPI_FAMILY_PC_APP
  *handle = ::LoadPackagedLibrary(wlibrary_filename.c_str(), 0);
#else
  // TODO: in most cases, the path name is a relative path and the behavior of the following line of code is undefined.
  *handle = ::LoadLibraryExW(wlibrary_filename.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#endif
  if (!*handle) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME,
                           FAIL,
                           "Loadibrary failed with error ",
                           error_code,
                           " - ",
                           std::system_category().message(error_code));
  }
  return Status::OK();
#else
  dlerror();  // clear any old error_str
  *handle = dlopen(library_filename.c_str(), RTLD_NOW | (global_symbols ? RTLD_GLOBAL : RTLD_LOCAL));
  char* error_str = dlerror();
  if (!*handle) {
    return Status(ONNXRUNTIME, FAIL, "Failed to load library " + library_filename + " with error: " + error_str);
  }
  return Status::OK();
#endif
}

Status OrtUnloadDynamicLibrary(void* handle) {
#ifdef _WIN32
  if (::FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME,
                           FAIL,
                           "FreeLibrary failed with error ",
                           error_code,
                           " - ",
                           std::system_category().message(error_code));
  }
  return Status::OK();
#else
  if (!handle) {
    return Status(ONNXRUNTIME, FAIL, "Got null library handle");
  }
  dlerror();  // clear any old error_str
  int retval = dlclose(handle);
  char* error_str = dlerror();
  if (retval != 0) {
    return Status(ONNXRUNTIME, FAIL, "Failed to unload library with error: " + std::string(error_str));
  }
  return Status::OK();
#endif
}

#ifdef _WIN32
namespace ort_dlfcn_win32 {
// adapted from https://github.com/dlfcn-win32 version 1.3.1.
// Simplified to only support finding symbols in libraries that were linked against.
// If ORT dynamically loads a custom ops library using RegisterCustomOpsLibrary[_V2] the handle from the library load
// is explicitly provided in the call to GetSymbolFromLibrary.
//
/* Load Psapi.dll at runtime, this avoids linking caveat */
bool OrtMyEnumProcessModules(HANDLE hProcess, HMODULE* lphModule, DWORD cb, LPDWORD lpcbNeeded) {
  using EnumProcessModulesFn = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
  static EnumProcessModulesFn EnumProcessModulesPtr = []() {
    EnumProcessModulesFn fn = nullptr;
    // Windows 7 and newer versions have K32EnumProcessModules in Kernel32.dll which is always pre-loaded
    HMODULE psapi = GetModuleHandleA("Kernel32.dll");
    if (psapi) {
      fn = (EnumProcessModulesFn)(LPVOID)GetProcAddress(psapi, "K32EnumProcessModules");
    }

    return fn;
  }();

  if (EnumProcessModulesPtr == nullptr) {
    return false;
  }

  return EnumProcessModulesPtr(hProcess, lphModule, cb, lpcbNeeded);
}

void* OrtSearchModulesForSymbol(const char* name) {
  HANDLE current_proc = GetCurrentProcess();
  DWORD size = 0;
  void* symbol = nullptr;

  // GetModuleHandle(NULL) only returns the current program file. So if we want to get ALL loaded module including
  // those in linked DLLs, we have to use EnumProcessModules().
  if (OrtMyEnumProcessModules(current_proc, nullptr, 0, &size) != false) {
    size_t num_handles = size / sizeof(HMODULE);
    std::unique_ptr<HMODULE[]> modules = std::make_unique<HMODULE[]>(num_handles);
    HMODULE* modules_ptr = modules.get();
    DWORD cb_needed = 0;
    if (OrtMyEnumProcessModules(current_proc, modules_ptr, size, &cb_needed) != 0 && size == cb_needed) {
      for (size_t i = 0; i < num_handles; i++) {
        symbol = GetProcAddress(modules[i], name);
        if (symbol != nullptr) {
          break;
        }
      }
    }
  }

  return symbol;
}
}  // namespace ort_dlfcn_win32
#endif  // ifdef _WIN32

Status OrtGetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) {
#ifdef _WIN32
  Status status = Status::OK();

  // global search to replicate dlsym RTLD_DEFAULT if handle is nullptr
  if (handle == nullptr) {
    *symbol = ort_dlfcn_win32::OrtSearchModulesForSymbol(symbol_name.c_str());
  } else {
    *symbol = ::GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name.c_str());
  }

  if (!*symbol) {
    const auto error_code = GetLastError();
    return ORT_MAKE_STATUS(ONNXRUNTIME,
                           FAIL,
                           "GetSymbol failed with error ",
                           error_code,
                           " - ",
                           std::system_category().message(error_code));
  }

  return status;
#else
  dlerror();  // clear any old error str

  // search global space if handle is nullptr.
  // value of RTLD_DEFAULT differs across posix platforms (-2 on macos, 0 on linux).
  handle = handle ? handle : RTLD_DEFAULT;
  *symbol = dlsym(handle, symbol_name.c_str());

  char* error_str = dlerror();
  if (error_str) {
    return Status(ONNXRUNTIME, FAIL, "Failed to get symbol " + symbol_name + " with error: " + error_str);
  }
  // it's possible to get a NULL symbol in our case when Schemas are not custom.
  return Status::OK();
#endif
}

}  // namespace onnxruntime
