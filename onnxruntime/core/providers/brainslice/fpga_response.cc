#include "core/providers/brainslice/fpga_response.h"
namespace onnxruntime {
namespace fpga {
FPGAResponse::FPGAResponse()
    : FPGAResponse(nullptr, 0) {
}

FPGAResponse::FPGAResponse(void* p_buffer,
                           size_t p_size,
                           FPGA_STATUS p_status)
    : FPGABuffer(p_buffer,
                 p_size),
      m_status(p_status) {
}

FPGAResponse::FPGAResponse(FPGAResponse&& p_other) {
  *this = std::move(p_other);
}

FPGAResponse& FPGAResponse::operator=(FPGAResponse&& p_other) {
  if (this != &p_other) {
    FPGABuffer::operator=(std::move(p_other));
    m_status = p_other.m_status;
  }

  return *this;
}

FPGAResponse::~FPGAResponse() {
}

FPGA_STATUS FPGAResponse::GetStatus() const {
  return m_status;
}

size_t FPGAResponse::GetResponseSize() const {
  return GetBufferSize();
}

}  // namespace fpga
}  // namespace onnxruntime
