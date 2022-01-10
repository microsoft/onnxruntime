#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor LogSoftmax(const tvm::te::Tensor& input, int64_t axis, const std::string& name = "logsoftmax");

}  // namespace tvm_codegen
}  // namespace onnxruntime
