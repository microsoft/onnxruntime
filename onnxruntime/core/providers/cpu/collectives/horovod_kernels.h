
#include "core/providers/common.h"
#include "core/framework/op_kernel.h"
#include <condition_variable>
#include <mutex>

namespace onnxruntime
{

class HorovodAllReduceOp: public OpKernel {
public:
 HorovodAllReduceOp(const OpKernelInfo& info) : OpKernel(info) {
   unique_name = "AllReduceNode_" + info.node().Name();
}

 Status Compute(OpKernelContext* context) const override;
private:
  std::string unique_name;
};

}