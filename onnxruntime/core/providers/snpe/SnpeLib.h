#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "core/common/logging/macros.h"
#include "core/common/logging/logging.h"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlError.hpp"

namespace onnxruntime {
namespace contrib {
namespace snpe {

enum BufferType {
  UNKNOWN = -1,
  ITENSOR,
  TF8,
  TF16,
  UINT8,
  FLOAT};

class SnpeLibRuntimeTarget {
 public:
  SnpeLibRuntimeTarget() {
    runtime_ =
#if defined(_M_X64)
        zdl::DlSystem::Runtime_t::CPU;
#else
        zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
#endif
  }

  explicit SnpeLibRuntimeTarget(const std::string& runtime) : SnpeLibRuntimeTarget() {
    Set(runtime);
  }

  zdl::DlSystem::Runtime_t Get() const { return runtime_; }

  void Set(const std::string& runtime) {
    if (!runtime.empty()) {
      if (runtime == "DSP" || runtime == "DSP_FIXED8_TF") {
        runtime_ = zdl::DlSystem::Runtime_t::DSP;
        return;
      }

      if (runtime == "CPU" || runtime == "CPU_FLOAT32") {
        runtime_ = zdl::DlSystem::Runtime_t::CPU;
        return;
      }

      if (runtime == "GPU" || runtime == "GPU_FLOAT32_16_HYBRID") {
        runtime_ = zdl::DlSystem::Runtime_t::GPU;
        return;
      }

      if (runtime == "GPU_FLOAT16") {
        runtime_ = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
        return;
      }

      if (runtime == "AIP_FIXED_TF" || runtime == "AIP_FIXED8_TF") {
        runtime_ = zdl::DlSystem::Runtime_t::AIP_FIXED_TF;
        return;
      }
    }
  }

  void Set(zdl::DlSystem::Runtime_t runtime) { runtime_ = runtime; }

  bool IsAvailable() const {
    zdl::DlSystem::RuntimeCheckOption_t runtime_check_option = zdl::DlSystem::RuntimeCheckOption_t::DEFAULT;
    // check availability, explicitly requiring unsignedpd support
    if (runtime_ == zdl::DlSystem::Runtime_t::DSP) {
      runtime_check_option = zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK;
    }
    return zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_, runtime_check_option);
  }

  std::string ToString() const {
    return std::string(zdl::DlSystem::RuntimeList::runtimeToString(runtime_));
  }

  ~SnpeLibRuntimeTarget() {}

 private:
  zdl::DlSystem::Runtime_t runtime_;
};

class SnpeRuntimeOptions {
 public:
  SnpeRuntimeOptions()
      : runtime_target_()
      , execution_priority_(zdl::DlSystem::ExecutionPriorityHint_t::NORMAL)
      , runtime_options_()
      , buffer_type_(BufferType::ITENSOR) {
  }

  explicit SnpeRuntimeOptions(const std::unordered_map<std::string, std::string>& options)
      : runtime_target_()
      , execution_priority_(zdl::DlSystem::ExecutionPriorityHint_t::NORMAL)
      , runtime_options_(options)
      , buffer_type_(BufferType::ITENSOR) {
      ParseOptions();
  }

  const SnpeLibRuntimeTarget& GetRuntimeTarget() const {
    return runtime_target_;
  }

  zdl::DlSystem::ExecutionPriorityHint_t GetExecutionPriority() const {
    return execution_priority_;
  }

  int GetBufferType() const {
    return buffer_type_;
  }

 private:
  void ParseOptions() {
    static const std::string OPT_RUNTIME = "runtime";
    static const std::string OPT_PRIORITY = "priority";
    static const std::string BUFFER_TYPE = "buffer_type";

    // Option - Runtime
    if (runtime_options_.find(OPT_RUNTIME) != runtime_options_.end()) {
      runtime_target_ = SnpeLibRuntimeTarget(runtime_options_[OPT_RUNTIME]);
      LOGS_DEFAULT(INFO) << "Located user specified runtime target: " << runtime_options_[OPT_RUNTIME];
    }

    // Option Priority
    if (runtime_options_.find(OPT_PRIORITY) != runtime_options_.end()) {
      if (runtime_options_[OPT_PRIORITY] == "low") {
        execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::LOW;
      } else if (runtime_options_[OPT_PRIORITY] == "normal") {
        execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::NORMAL;
      } else {
        LOGS_DEFAULT(INFO) << "Invalid execution priority, defaulting to LOW";
        execution_priority_ = zdl::DlSystem::ExecutionPriorityHint_t::LOW;
      }

      LOGS_DEFAULT(INFO) << "Located user specified execution priority " << runtime_options_[OPT_PRIORITY];
    }

    // buffer type
    if (runtime_options_.find(BUFFER_TYPE) != runtime_options_.end()) {
      if (runtime_options_[BUFFER_TYPE] == "TF8") {
        buffer_type_ = BufferType::TF8;
      } else if (runtime_options_[BUFFER_TYPE] == "TF16") {
        buffer_type_ = BufferType::TF16;
      } else if (runtime_options_[BUFFER_TYPE] == "ITENSOR") {
        buffer_type_ = BufferType::ITENSOR;
      } else if (runtime_options_[BUFFER_TYPE] == "UINT8") {
        buffer_type_ = BufferType::UINT8;
      } else if (runtime_options_[BUFFER_TYPE] == "FLOAT") {
        buffer_type_ = BufferType::FLOAT;
      } else {
        LOGS_DEFAULT(ERROR) << "Invalid buffer type: " << runtime_options_[BUFFER_TYPE];
        buffer_type_ = BufferType::UNKNOWN;
      }
    }
  }

 private:
  SnpeLibRuntimeTarget runtime_target_;
  zdl::DlSystem::ExecutionPriorityHint_t execution_priority_;
  std::unordered_map<std::string, std::string> runtime_options_;
  std::string udo_folder_;
  std::vector<std::string> udo_names_;
  int buffer_type_;
};

struct UserBufferAttribute {
 public:
  UserBufferAttribute(size_t size,
                      const std::vector<size_t>& buffer_strides,
                      zdl::DlSystem::UserBufferEncoding* const buffer_encoding) :
      buffer_size(size),
      strides(buffer_strides),
      user_buffer_encoding(buffer_encoding)
  {}

  size_t buffer_size;
  std::vector<size_t> strides;
  zdl::DlSystem::UserBufferEncoding* user_buffer_encoding;
};

class SnpeLib {
 public:
  SnpeLib() : buffer_type_(BufferType::ITENSOR) {}
  ~SnpeLib() {}

  bool SnpeProcess(const unsigned char* input,
                   size_t input_size,
                   unsigned char* output,
                   size_t output_size,
                   const std::unordered_map<std::string, size_t>& output_names_index);
  bool SnpeProcessMultipleOutput(const unsigned char* input,
                                 size_t input_size,
                                 size_t output_number,
                                 unsigned char* outputs[],
                                 size_t output_sizes[],
                                 const std::unordered_map<std::string, size_t>& output_names_index);
  bool SnpeProcessMultipleInputsMultipleOutputs(const unsigned char** inputs,
                                                const size_t* input_sizes,
                                                size_t input_number,
                                                unsigned char** outputs,
                                                const size_t* output_sizes,
                                                size_t output_number,
                                                const std::unordered_map<std::string, size_t>& output_names_index);
  bool SnpeProcessWithUserBuffer(const std::vector<std::string>& input_names,
                                 const unsigned char** inputs,
                                 size_t input_number,
                                 unsigned char** outputs,
                                 const std::unordered_map<std::string, size_t>& output_names_index);

  bool CheckInputsSize(const std::vector<std::string>& input_tensor_names,
                       const std::vector<int64_t>& input_sizes);
  bool InitializeSnpe(zdl::DlContainer::IDlContainer* container,
                      const std::vector<std::string>& output_tensor_names,
                      const std::vector<std::string>& input_tensor_names,
                      const std::vector<int64_t>& input_sizes,
                      const SnpeRuntimeOptions& settings = SnpeRuntimeOptions());
  bool Initialize(const char* dlcPath,
                  const std::vector<std::string>& output_layer_names,
                  const std::vector<std::string>& input_layer_names,
                  const std::vector<int64_t>& input_sizes,
                  const SnpeRuntimeOptions& settings = SnpeRuntimeOptions());
  bool Initialize(const unsigned char* dlcData,
                  size_t size,
                  const std::vector<std::string>& output_layer_names,
                  const std::vector<std::string>& input_layer_names,
                  const std::vector<int64_t>& input_sizes,
                  const SnpeRuntimeOptions& settings = SnpeRuntimeOptions());

  bool SetupUserBufferAttribute(const std::string& name);
  bool SetupUserBufferAttributes(const std::vector<std::string>& tensor_names);
  bool SetupInputTensors(const std::vector<std::string>& input_tensor_names);

 private:
  const char* GetSnpeErrorString() {
    return zdl::DlSystem::getLastErrorString();
  }

  int RegisterUDOs(const std::string udo_dir, const std::vector<std::string>& udo_file_names) {
    int udos_registered = 0;

    for (auto udo_file = udo_file_names.begin(); udo_file != udo_file_names.end(); ++udo_file) {
      std::string full_path = udo_dir + "/" + *udo_file;
      bool result = zdl::SNPE::SNPEFactory::addOpPackage(full_path);
      if (result) {
        ++udos_registered;
      } else {
        LOGS_DEFAULT(ERROR) << "Failed to register SNPE UDO library: " << full_path << " :" << GetSnpeErrorString();
      }
    }
    return udos_registered;
  }

  std::unique_ptr<zdl::SNPE::SNPE> snpe_;
  std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> input_tensors_;
  zdl::DlSystem::TensorMap input_tensor_map_;

  int buffer_type_;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_input_buffers_;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_output_buffers_;
  std::vector<std::unique_ptr<zdl::DlSystem::UserBufferEncoding>> user_buffer_encoding_;
  std::unordered_map<std::string, UserBufferAttribute> user_buffer_attr_table_;
};

std::unique_ptr<SnpeLib> SnpeLibFactory(const unsigned char* dlc_data,
                                        size_t size,
                                        const std::unordered_map<std::string, std::string>& options,
                                        const std::vector<std::string>& output_layer_names,
                                        const std::vector<std::string>& input_layer_names,
                                        const std::vector<int64_t>& input_sizes,
                                        int& buffer_type);

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime