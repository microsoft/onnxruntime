#include "core/providers/brainslice/fpga_helper.h"
namespace onnxruntime {
namespace fpga {
uint32_t
FPGAUtil::FlipUint32(const uint32_t p_in) {
  uint32_t temp = 0;

  temp += ((p_in >> 0) & 0xff) << 24;
  temp += ((p_in >> 8) & 0xff) << 16;
  temp += ((p_in >> 16) & 0xff) << 8;
  temp += ((p_in >> 24) & 0xff) << 0;

  return temp;
}
}  // namespace fpga
}  // namespace onnxruntime
