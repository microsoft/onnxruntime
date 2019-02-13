#include "core/providers/brainslice/fpga_request.h"

namespace onnxruntime {
namespace fpga {
FPGARequest::FPGARequest() : FPGARequest(0, nullptr, 0) {
}

FPGARequest::FPGARequest(FPGARequestType request_type,
                         void* buffer,
                         size_t size) : FPGABuffer(buffer, size),
                                        request_type_(request_type),
                                        stream_(*this) {
}

FPGARequestType FPGARequest::GetRequestType() const {
  return request_type_;
}

size_t FPGARequest::GetRequestSize() const {
  return stream_.GetCurrentOffset();
}

FPGABufferStream& FPGARequest::GetStream() {
  return stream_;
}

}  // namespace fpga
}  // namespace onnxruntime
