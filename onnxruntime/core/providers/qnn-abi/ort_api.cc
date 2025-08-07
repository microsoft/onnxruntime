// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/ort_api.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include <gsl/gsl>

namespace onnxruntime {

namespace {

OrtNodeUnitIODef ParseOrtValueInfo(const OrtValueInfo* io,
                                   std::optional<OrtNodeUnitIODef::QuantParam> quant_param,
                                   const OrtApi& ort_api) {
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

  return OrtNodeUnitIODef{name, elem_type, shape, quant_param};
}

std::vector<OrtNodeUnitIODef> GetQDQIODefs(const OrtNode* target_node,
                                           const QDQ::OrtNodeGroup& node_group,
                                           bool is_input,
                                           const OrtApi& ort_api) {
  const auto& dq_or_q_nodes = is_input ? node_group.dq_nodes : node_group.q_nodes;

  size_t num_ios = 0;
  std::vector<const OrtValueInfo*> target_node_ios;
  if (is_input) {
    ort_api.Node_GetNumInputs(target_node, &num_ios);
    target_node_ios.resize(num_ios);
    ort_api.Node_GetInputs(target_node, target_node_ios.data(), target_node_ios.size());
  } else {
    ort_api.Node_GetNumOutputs(target_node, &num_ios);
    target_node_ios.resize(num_ios);
    ort_api.Node_GetOutputs(target_node, target_node_ios.data(), target_node_ios.size());
  }

  // Find all the quantized IO defs and indices (for the input/output of the target node).
  std::unordered_map<size_t, OrtNodeUnitIODef> quantized_io_defs;
  quantized_io_defs.reserve(num_ios);

  for (size_t io_idx = 0; io_idx < num_ios; ++io_idx) {
    const OrtNode* node = nullptr;
    if (is_input) {
      ort_api.ValueInfo_GetValueProducer(target_node_ios[io_idx], &node, nullptr);
    } else {
      // TODO: Not sure whether functionally identical to old implementation.
      size_t num_consumers = 0;
      ort_api.ValueInfo_GetValueNumConsumers(target_node_ios[io_idx], &num_consumers);
      if (num_consumers != 1) {
        continue;
      }

      std::vector<const OrtNode*> consumers(num_consumers);
      std::vector<int64_t> input_indices(num_consumers);
      ort_api.ValueInfo_GetValueConsumers(target_node_ios[io_idx],
                                          consumers.data(),
                                          input_indices.data(),
                                          num_consumers);
      node = consumers[0];
    }

    // If we cannot find the node index in the DQ/Q nodes, this is not a quantized input/output.
    if (std::find(dq_or_q_nodes.cbegin(), dq_or_q_nodes.cend(), node) == dq_or_q_nodes.cend()) {
      continue;
    }

    size_t num_node_inputs = 0;
    ort_api.Node_GetNumInputs(node, &num_node_inputs);
    std::vector<const OrtValueInfo*> node_inputs(num_node_inputs);
    ort_api.Node_GetInputs(node, node_inputs.data(), node_inputs.size());

    // Get the Q/DQ axis attribute if available.
    std::optional<int64_t> axis = OrtNodeAttrHelper(ort_api, *node).GetInt64("axis");

    // Quantization scale and zp are always the input[1, 2].
    OrtNodeUnitIODef::QuantParam quant_param{node_inputs[1], num_node_inputs == 3 ? node_inputs[2] : nullptr, axis};

    OrtNodeUnitIODef io_def;
    if (is_input) {
      // DQ node, using input[0, 1, 2].
      io_def = ParseOrtValueInfo(node_inputs[0], quant_param, ort_api);
    } else {
      // Q node, using output[0], input[1, 2].
      size_t num_node_outputs = 0;
      ort_api.Node_GetNumOutputs(node, &num_node_outputs);
      std::vector<const OrtValueInfo*> node_outputs(num_node_outputs);
      ort_api.Node_GetOutputs(node, node_outputs.data(), node_outputs.size());
      io_def = ParseOrtValueInfo(node_outputs[0], quant_param, ort_api);
    }

    quantized_io_defs.insert({io_idx, io_def});
  }

  // Construct IO defs for this QDQ node group.
  std::vector<OrtNodeUnitIODef> io_defs;
  io_defs.reserve(num_ios);
  for (size_t io_idx = 0; io_idx < num_ios; ++io_idx) {
    // If we can find the NodeUnitIODef for this index, this is a quantized input/output.
    if (quantized_io_defs.find(io_idx) != quantized_io_defs.end()) {
      io_defs.push_back(std::move(quantized_io_defs.at(io_idx)));
    } else {
      // This is a regular input.
      io_defs.push_back(ParseOrtValueInfo(target_node_ios[io_idx], std::nullopt, ort_api));
    }
  }

  return io_defs;
}

}  // namespace

OrtNodeUnit::OrtNodeUnit(const OrtNode* node, const OrtApi& ort_api) : target_node_(node), type_(Type::SingleNode) {
  InitForSingleNode(ort_api);
}

OrtNodeUnit::OrtNodeUnit(const OrtGraph* /* graph */, const QDQ::OrtNodeGroup& node_group, const OrtApi& ort_api)
    : dq_nodes_(node_group.dq_nodes),
      target_node_(node_group.target_node),
      redundant_clip_node_(node_group.redundant_clip_node ? node_group.redundant_clip_node : nullptr),
      q_nodes_(node_group.q_nodes),
      type_(Type::QDQGroup),
      inputs_(GetQDQIODefs(target_node_, node_group, true, ort_api)),
      outputs_(GetQDQIODefs((redundant_clip_node_ ? redundant_clip_node_ : target_node_), node_group, false, ort_api)) {
}

void OrtNodeUnit::InitForSingleNode(const OrtApi& ort_api) {
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  ort_api.Node_GetNumInputs(target_node_, &num_inputs);
  ort_api.Node_GetNumOutputs(target_node_, &num_outputs);

  std::vector<const OrtValueInfo*> inputs_data(num_inputs);
  std::vector<const OrtValueInfo*> outputs_data(num_outputs);
  ort_api.Node_GetInputs(target_node_, inputs_data.data(), inputs_data.size());
  ort_api.Node_GetOutputs(target_node_, outputs_data.data(), outputs_data.size());

  const char* op_type = nullptr;
  ort_api.Node_GetOperatorType(target_node_, &op_type);

  if (std::string(op_type) == "DequantizeLinear") {
    std::optional<int64_t> axis = OrtNodeAttrHelper(ort_api, *target_node_).GetInt64("axis");
    OrtNodeUnitIODef::QuantParam quant_param{inputs_data[1], num_inputs == 3 ? inputs_data[2] : nullptr, axis};
    inputs_.push_back(ParseOrtValueInfo(inputs_data[0], quant_param, ort_api));
    outputs_.push_back(ParseOrtValueInfo(outputs_data[0], std::nullopt, ort_api));
  } else if (std::string(op_type) == "QuantizeLinear") {
    std::optional<int64_t> axis = OrtNodeAttrHelper(ort_api, *target_node_).GetInt64("axis");
    OrtNodeUnitIODef::QuantParam quant_param{inputs_data[1], num_inputs == 3 ? inputs_data[2] : nullptr, axis};
    inputs_.push_back(ParseOrtValueInfo(inputs_data[0], std::nullopt, ort_api));
    outputs_.push_back(ParseOrtValueInfo(outputs_data[0], quant_param, ort_api));
  } else {
    inputs_.reserve(num_inputs);
    for (size_t idx = 0; idx < num_inputs; ++idx) {
      const OrtValueInfo* io = inputs_data[idx];
      // Nullptr indicates optional.
      OrtNodeUnitIODef io_def = io == nullptr ? OrtNodeUnitIODef{"", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, {}}
                                              : ParseOrtValueInfo(io, std::nullopt, ort_api);
      inputs_.push_back(io_def);
    }

    outputs_.reserve(num_outputs);
    for (size_t idx = 0; idx < num_outputs; ++idx) {
      const OrtValueInfo* io = outputs_data[idx];
      // Not sure whether output would be optional.
      OrtNodeUnitIODef io_def = io == nullptr ? OrtNodeUnitIODef{"", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, {}}
                                              : ParseOrtValueInfo(io, std::nullopt, ort_api);
      outputs_.push_back(io_def);
    }
  }
}

size_t OrtNodeUnit::GetInputEdgesCount(const OrtApi& ort_api) const {
  auto count_edges = [&ort_api](const OrtNode* target_node) {
    size_t num_inputs = 0;
    ort_api.Node_GetNumInputs(target_node, &num_inputs);

    std::vector<const OrtValueInfo*> inputs(num_inputs);
    ort_api.Node_GetInputs(target_node, inputs.data(), inputs.size());

    size_t num_actual_inputs = 0;
    for (const OrtValueInfo* input : inputs) {
      if (input == nullptr) {
        continue;
      }

      const OrtNode* producer_node = nullptr;
      ort_api.ValueInfo_GetValueProducer(input, &producer_node, nullptr);
      if (producer_node) {
        num_actual_inputs++;
      }
    }

    return num_actual_inputs;
  };

  if (type_ == Type::SingleNode) {
    return count_edges(target_node_);
  }

  size_t edges = std::accumulate(dq_nodes_.cbegin(),
                                 dq_nodes_.cend(),
                                 size_t(0),
                                 [&count_edges](size_t acc, const OrtNode* node) { return acc + count_edges(node); });
  return edges + count_edges(target_node_) - dq_nodes_.size();
}

std::vector<const OrtNode*> OrtNodeUnit::GetOutputNodes(const OrtApi& ort_api) const {
  auto get_consumers = [&ort_api](const OrtNode* target_node) {
    std::vector<const OrtNode*> target_consumers;

    size_t num_outputs = 0;
    ort_api.Node_GetNumOutputs(target_node, &num_outputs);

    std::vector<const OrtValueInfo*> outputs(num_outputs);
    ort_api.Node_GetOutputs(target_node, outputs.data(), outputs.size());

    for (const OrtValueInfo* output : outputs) {
      if (output == nullptr) {
        continue;
      }

      size_t num_consumers = 0;
      ort_api.ValueInfo_GetValueNumConsumers(output, &num_consumers);

      std::vector<const OrtNode*> consumers(num_consumers);
      std::vector<int64_t> input_indices(num_consumers);
      ort_api.ValueInfo_GetValueConsumers(output, consumers.data(), input_indices.data(), num_consumers);

      target_consumers.insert(target_consumers.end(), consumers.begin(), consumers.end());
    }

    return target_consumers;
  };

  const OrtNode* output_producer = redundant_clip_node_ ? redundant_clip_node_ : target_node_;
  std::vector<const OrtNode*> output_nodes;

  for (const OrtNode* output_node : get_consumers(output_producer)) {
    if (std::find(q_nodes_.cbegin(), q_nodes_.cend(), output_node) != q_nodes_.cend()) {
      std::vector<const OrtNode*> q_output_nodes = get_consumers(output_node);
      output_nodes.insert(output_nodes.end(), q_output_nodes.begin(), q_output_nodes.end());
    } else {
      output_nodes.push_back(output_node);
    }
  }

  return output_nodes;
}

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
  return api_node_attr->attr_proto.s();
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

OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& ort_api,
                                          const OrtSessionOptions& session_options,
                                          const std::string& config_key,
                                          const std::string& default_val,
                                          /*out*/ std::string& config_val) {
  const char* config_key_cstr = config_key.c_str();

  int has_config = 0;
  RETURN_IF_ERROR(ort_api.HasSessionConfigEntry(&session_options, config_key_cstr, &has_config));

  if (has_config != 1) {
    config_val = default_val;
    return nullptr;
  }

  size_t size = 0;
  RETURN_IF_ERROR(ort_api.GetSessionConfigEntry(&session_options, config_key_cstr, nullptr, &size));

  config_val.resize(size);
  RETURN_IF_ERROR(ort_api.GetSessionConfigEntry(&session_options, config_key_cstr, config_val.data(), &size));
  config_val.resize(size - 1);  // remove the terminating '\0'

  return nullptr;
}

std::string GetProviderOptionPrefix(const std::string& provider_name) {
  std::string key_prefix = "ep.";
  key_prefix += utils::GetLowercaseString(provider_name);
  key_prefix += ".";

  return key_prefix;
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
