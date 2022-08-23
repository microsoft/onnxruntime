#include "my_execution_provider.h"

namespace onnxruntime {

class Add : public onnxruntime::OpKernel {
 public:
  Add(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;
};

}
