#pragma once
#include <tvm/tvm.h>

#include <string>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor LogSoftmax(const tvm::Tensor& input, int64_t axis, const std::string& name = "logsoftmax");

}  // namespace tvm_codegen
}  // namespace onnxruntime
