// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/brainslice/fpga_util.h"

namespace onnxruntime {
namespace fpga {

struct FPGAInfo {
  //Ip address of FPGA device
  uint32_t ip;
  //Is the FPGA already configured.
  bool need_configure;
  //Following field only meaningful when need_configure is True.
  //Path to the instruction bin file
  const char* inst_file;
  //Path to the data bin file
  const char* data_file;
  //Path to the schema bin file
  const char* schema_file;
};

constexpr int msg_header_bytes = 1024;

class FPGAHandle {
 public:
  FPGAHandle(FPGAInfo info);
  virtual ~FPGAHandle() {}

  const BrainSlice_Parameters GetParameters() const {
    return parameters_;
  }

  Status GetCapabilities(BS_Capabilities* capabilities) const;

  Status LoadMatrix(const std::vector<half_float::half>& matrix, const int rows, const int cols,
                    const int matix_addr, const bool row_major, const ISA_Mem mem_type, size_t* num_tiles = nullptr) const;

  Status LoadVector(const std::vector<half_float::half>& vector, const int vec_addr, const ISA_Mem mem_type, size_t* num_tiles = nullptr) const;

  Status SendSync(std::function<int32_t(void*, size_t*)> prepare_request, std::function<int32_t(void*, size_t)> process_response) const;

  // To load a firmware to brainslice, three files are needed:
  // 1. the instructions bin file
  // 2. the data bin file
  // 3. the schema bin file
  Status LoadFirmware(const std::string& inst_file,
                      const std::string& data_file,
                      const std::string& schema_file) const;

  Status LoadFirmware(std::vector<uint32_t>&& inst,
                      std::vector<uint32_t>&& data,
                      std::vector<uint64_t>&& schema) const;

 private:
  BrainSlice_Parameters parameters_;
  uint32_t ip_;
  size_t max_request_size_;
};
}  // namespace fpga
}  // namespace onnxruntime
