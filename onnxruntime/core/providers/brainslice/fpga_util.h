#pragma once
#include "FPGACoreLib.h"
#include "CommonFunctions_client.h"
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_request.h"
#include "3rdparty/half.hpp"
#include <mutex>

namespace onnxruntime {
namespace fpga {
class FPGAUtil {
 public:
  virtual ~FPGAUtil();

  static FPGAUtil& Instance();

  Status IsDeviceReady() {
    return status_ == Initialized ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL, "FPGA device initialization failed");
  }

  enum FPGA_Status : int {
    NotInitialized = 0,
    Initialized = 1,
    InitializeFailed = 3
  };

  Status GetCapabilities(const uint32_t p_dstIP, BS_Capabilities* out);
  Status GetParameters(const uint32_t p_dstIP, BrainSlice_Parameters* out);

  Status SendSync(const uint32_t p_dstIP, std::function<int32_t(void*, size_t*)> prepare_request, std::function<int32_t(void*, size_t)> process_response);

  size_t GetBufferSize() { return max_buffer_size_; }

 private:
  FPGAUtil();

  static Status InitFPGA(std::unique_ptr<FPGAUtil>* handle);

  FPGA_HANDLE handle_;

  FPGA_Status status_;

  std::mutex fpga_mutex_;

  size_t max_buffer_size_;

  std::string error_message_;
};
}  // namespace fpga
}  // namespace onnxruntime
