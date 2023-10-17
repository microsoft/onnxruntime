#include "string_concat.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/common/common.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    StringConcat,
    20,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<std::string>()),
    StringConcat);

Status StringConcat::Compute(OpKernelContext* context) const {
  ProcessBroadcastSpanFuncs broadcast_funcs{
      [](BroadcastHelper& broadcast_helper) {
        auto x = broadcast_helper.ScalarInput0<std::string>();
        auto Y = broadcast_helper.SpanInput1<std::string>();
        auto output = broadcast_helper.OutputSpan<std::string>();
        std::transform(Y.begin(), Y.end(), output.begin(),
                       [&x](const std::string& y) {
                         return x + y;
                       });
      },
      [](BroadcastHelper& broadcast_helper) {
        auto X = broadcast_helper.SpanInput0<std::string>();
        auto y = broadcast_helper.ScalarInput1<std::string>();
        auto output = broadcast_helper.OutputSpan<std::string>();
        std::transform(X.begin(), X.end(), output.begin(),
                       [&y](const std::string& x) {
                         return x + y;
                       });
      },
      [](BroadcastHelper& broadcast_helper) {
        auto X = broadcast_helper.SpanInput0<std::string>();
        auto Y = broadcast_helper.SpanInput1<std::string>();
        auto output = broadcast_helper.OutputSpan<std::string>();
        std::transform(X.begin(), X.end(), Y.begin(), output.begin(),
                       [](const std::string& x, const std::string& y) {
                         return x + y;
                       });
      }};
  UntypedBroadcastTwo(*context, broadcast_funcs);
  return Status::OK();
}

}  // namespace onnxruntime
