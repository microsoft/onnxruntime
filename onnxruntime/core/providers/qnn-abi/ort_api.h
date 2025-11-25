// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <functional>
#include <gsl/gsl>
#include <memory>
#include <string>
#include <vector>

#include <SafeInt.hpp>

// This compilation unit (ort_api.h/.cc) encapsulates the interface between the EP and ORT in a manner
// that allows QNN EP to built either as a static library or a dynamic shared library.
// The preprocessor macro `BUILD_QNN_EP_STATIC_LIB` is defined and set to 1 if QNN EP
// is built as a static library.
// Includes when building QNN EP as a shared library
// #include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

#include "core/session/onnxruntime_c_api.h"

#include "core/common/inlined_containers.h"
#include "core/common/float16.h"
#include "core/framework/int4.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

namespace onnxruntime {

#define MAKE_FAIL(msg) Ort::Status(msg, ORT_FAIL)
#define MAKE_EP_FAIL(msg) Ort::Status(msg, ORT_EP_FAIL)

#define RETURN_IF(cond, msg)      \
  do {                            \
    if ((cond)) {                 \
      return MAKE_EP_FAIL((msg)); \
    }                             \
  } while (0)

#define RETURN_IF_NOT(cond, msg) \
  RETURN_IF(!(cond), msg)

#define RETURN_IF_ERROR(fn)     \
  do {                          \
    Ort::Status _status = (fn); \
    if (!_status.IsOK()) {      \
      return _status;           \
    }                           \
  } while (0)

#define RETURN_IF_NOT_OK(fn)    \
  do {                          \
    Ort::Status _status = (fn); \
    if (!_status.IsOK()) {      \
      return _status.release(); \
    }                           \
  } while (0)

#define RETURN_IF_NOT_NULL(fn) \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_DEFAULT_IF_API_FAIL(ort_api_fn_call, ort_api, ret_val) \
  do {                                                                \
    if (OrtStatus* _status = (ort_api_fn_call)) {                     \
      (ort_api).ReleaseStatus(_status);                               \
      return (ret_val);                                               \
    }                                                                 \
  } while (0)

// QNN-EP COPY START
// Below are macors copied from core/common/common.h directly.
#ifdef _WIN32
#define ORT_UNUSED_PARAMETER(x) (x)
#else
#define ORT_UNUSED_PARAMETER(x) (void)(x)
#endif

// Macros to disable the copy and/or move ctor and assignment methods
// These are usually placed in the private: declarations for a class.
#define ORT_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define ORT_DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  ORT_DISALLOW_COPY(TypeName);                     \
  ORT_DISALLOW_ASSIGNMENT(TypeName)

#define ORT_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete

#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  ORT_DISALLOW_MOVE(TypeName)
// QNN-EP COPY END

// Since ORT_TSTR expands to wstring and string on WIN32 and non-WIN32, respectively, this macro provides convenient
// usage to convert std::filesystem::path accordingly.
#ifdef _WIN32
#define FILEPATH_TO_STRING(filepath) (filepath).wstring();
#else
#define FILEPATH_TO_STRING(filepath) (filepath).string();
#endif

// QNN-EP COPY START
// Below are GSL utilities copied from core/common/span_utils.h directly.
template <class U, class T>
[[nodiscard]] inline gsl::span<U> ReinterpretAsSpan(gsl::span<T> src) {
  // adapted from gsl-lite span::as_span():
  // https://github.com/gsl-lite/gsl-lite/blob/4720a2980a30da085b4ddb4a0ea2a71af7351a48/include/gsl/gsl-lite.hpp#L4102-L4108
  Expects(src.size_bytes() % sizeof(U) == 0);
  return gsl::span<U>(reinterpret_cast<U*>(src.data()), src.size_bytes() / sizeof(U));
}

// Below are constants copied from core/graph/constants.h directly.
constexpr const char* kOnnxDomain = "";
constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kMSInternalNHWCDomain = "com.ms.internal.nhwc";
// QNN-EP COPY END

class OrtLoggingManager {
 public:
  static const Ort::Logger& GetDefaultLogger() {
    return GetLoggerInstance();
  }

  static bool HasDefaultLogger() {
    return GetLoggerPtr() != nullptr;
  }

  static void SetDefaultLogger(const OrtLogger* default_logger) {
    GetLoggerPtr() = default_logger;
  }

 private:
  OrtLoggingManager() = delete;

  static const OrtLogger*& GetLoggerPtr() {
    static const OrtLogger* default_logger_ = nullptr;
    return default_logger_;
  }

  static const Ort::Logger& GetLoggerInstance() {
    static const Ort::Logger ort_logger_ = Ort::Logger(GetLoggerPtr());
    return ort_logger_;
  }
};

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

// Helper to release Ort one or more objects obtained from the public C API at the end of their scope.
template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
      : objects_(object_ptr), count_(1), release_func_(release_func) {}

  DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
      : objects_(objects), count_(count), release_func_(release_func) {}

  ~DeferOrtRelease() {
    if (objects_ != nullptr && count_ > 0) {
      for (size_t i = 0; i < count_; ++i) {
        if (objects_[i] != nullptr) {
          release_func_(objects_[i]);
          objects_[i] = nullptr;
        }
      }
    }
  }
  T** objects_ = nullptr;
  size_t count_ = 0;
  std::function<void(T*)> release_func_ = nullptr;
};

namespace QDQ {

// Define NodeGroup structure similar to the one in shared/utils.h
struct OrtNodeGroup {
  std::vector<const OrtNode*> dq_nodes;
  std::vector<const OrtNode*> q_nodes;
  const OrtNode* target_node;
  const OrtNode* redundant_clip_node{nullptr};
};

}  // namespace QDQ

struct OrtNodeUnitIODef {
  struct QuantParam {
    const OrtValueInfo* scale;
    const OrtValueInfo* zero_point{nullptr};
    std::optional<int64_t> axis{std::nullopt};
  };

  std::string name;
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
  std::optional<QuantParam> quant_param;

  bool Exists() const noexcept { return !name.empty(); }
};

class OrtNodeUnit {
 public:
  // NodeUnit type
  enum class Type : uint8_t {
    SingleNode,  // The NodeUnit contains a single node
    QDQGroup,    // The NodeUnit contain a QDQ group of nodes, such as "DQ->Sigmoid->Q"
  };

 public:
  explicit OrtNodeUnit(const OrtNode* node, const OrtApi& ort_api);
  explicit OrtNodeUnit(const OrtGraph* graph, const QDQ::OrtNodeGroup& node_group, const OrtApi& ort_api);

  Type UnitType() const noexcept { return type_; }

  const std::vector<OrtNodeUnitIODef>& Inputs() const noexcept { return inputs_; }
  const std::vector<OrtNodeUnitIODef>& Outputs() const noexcept { return outputs_; }

  std::string Domain() const noexcept { return Ort::ConstNode(target_node_).GetDomain(); }
  std::string OpType() const noexcept { return Ort::ConstNode(target_node_).GetOperatorType(); }
  std::string Name() const noexcept { return Ort::ConstNode(target_node_).GetName(); }
  int SinceVersion() const noexcept { return Ort::ConstNode(target_node_).GetSinceVersion(); }
  // Align NodeUnit to name as Index although returning Id since index is inaccessible.
  size_t Index() const noexcept { return Ort::ConstNode(target_node_).GetId(); }

  const OrtNode& GetNode() const noexcept { return *target_node_; }
  const OrtNode* GetRedundantClipNode() const noexcept { return redundant_clip_node_; }
  const std::vector<const OrtNode*>& GetDQNodes() const noexcept { return dq_nodes_; }
  const std::vector<const OrtNode*>& GetQNodes() const noexcept { return q_nodes_; }
  std::vector<const OrtNode*> GetAllNodesInGroup() const noexcept {
    std::vector<const OrtNode*> all_nodes = dq_nodes_;
    all_nodes.push_back(target_node_);
    if (redundant_clip_node_) {
      all_nodes.push_back(redundant_clip_node_);
    }
    all_nodes.reserve(all_nodes.size() + q_nodes_.size());
    for (auto& n : q_nodes_)
      all_nodes.push_back(n);
    return all_nodes;
  }

  size_t GetInputEdgesCount(const OrtApi& ort_api) const;
  std::vector<const OrtNode*> GetOutputNodes(const OrtApi& ort_api) const;

 private:
  // // Initialization for a NodeUnit that contains a single node
  OrtStatus* InitForSingleNode(const OrtApi& ort_api);

  const std::vector<const OrtNode*> dq_nodes_;  // dq nodes for this NodeUnit, not necessarily all inputs
  const OrtNode* target_node_;
  const OrtNode* redundant_clip_node_ = nullptr;  // Optional redundant clip node for the QDQ group, nullptr if not present.
  const std::vector<const OrtNode*> q_nodes_;     // q-nodes for this NodeUnit. not necessarily all outputs
  const Type type_;

  std::vector<OrtNodeUnitIODef> inputs_;
  std::vector<OrtNodeUnitIODef> outputs_;
};

/**
 * Wrapping onnxruntime::Node for retrieving attribute values
 */

class OrtNodeAttrHelper {
 public:
  explicit OrtNodeAttrHelper(const OrtNode& node);

  // Get the attributes from the target node of the node_unit
  explicit OrtNodeAttrHelper(const OrtNodeUnit& node_unit);

  /*
   * Get with default
   */
  float Get(const std::string& key, float def_val) const;
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;

  int64_t Get(const std::string& key, int64_t def_val) const;
  std::vector<int64_t> Get(const std::string& key, const std::vector<int64_t>& def_val) const;

  std::string Get(const std::string& key, std::string def_val) const;
  std::vector<std::string> Get(const std::string& key, const std::vector<std::string>& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to int32_t
  int32_t Get(const std::string& key, int32_t def_val) const;
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to uint32_t
  uint32_t Get(const std::string& key, uint32_t def_val) const;
  std::vector<uint32_t> Get(const std::string& key, const std::vector<uint32_t>& def_val) const;

  /*
   * Get without default.
   */
  std::optional<float> GetFloat(const std::string& key) const;
  std::optional<std::vector<float>> GetFloats(const std::string& key) const;

  std::optional<int64_t> GetInt64(const std::string& key) const;
  std::optional<std::vector<int64_t>> GetInt64s(const std::string& key) const;

  std::optional<std::string> GetString(const std::string& key) const;

  bool HasAttr(const std::string& key) const;

 private:
  const OrtNode& node_;
};

OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& ort_api,
                                          const OrtSessionOptions& session_options,
                                          const std::string& config_key,
                                          const std::string& default_val,
                                          /*out*/ std::string& config_val);

std::basic_string<ORTCHAR_T> GetModelPathString(const OrtGraph* graph, const OrtApi& ort_api);

/**
 * Returns a lowercase version of the input string.
 * /param str The string to lowercase.
 * /return The lowercased string.
 */
inline std::string GetLowercaseString(std::string str) {
  // https://en.cppreference.com/w/cpp/string/byte/tolower
  // The behavior of tolower from <cctype> is undefined if the argument is neither representable as unsigned char
  // nor equal to EOF. To use tolower safely with a plain char (or signed char), the argument must be converted to
  // unsigned char.
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return str;
}

// Refer to OrtSessionOptions::GetProviderOptionPrefix.
std::string GetProviderOptionPrefix(const std::string& provider_name);

/// @brief Gets the path of directory containing the dynamic library that contains the address.
/// @param address An address of a function or variable in the dynamic library.
/// @return The path of the directory containing the dynamic library, or an empty string if the path cannot be determined.
std::basic_string<ORTCHAR_T> GetDynamicLibraryLocationByAddress(const void* address);

// QNN-EP COPY START
// Below implementations are directly copied from "core/platform/posix/env.cc" and "core/platform/windows/env.cc"
// with few modifications to eliminate additional dependencies.
std::basic_string<ORTCHAR_T> OrtGetRuntimePath();

Ort::Status OrtLoadDynamicLibrary(const std::basic_string<ORTCHAR_T>& wlibrary_filename,
                                  bool global_symbols,
                                  void** handle);

Ort::Status OrtUnloadDynamicLibrary(void* handle);

Ort::Status OrtGetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol);

Ort::Status ReadFileIntoBuffer(const ORTCHAR_T* file_path, int64_t offset, size_t length, gsl::span<char> buffer);
// QNN-EP COPY END

}  // namespace onnxruntime
