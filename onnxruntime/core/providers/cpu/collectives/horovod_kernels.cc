#include "core/graph/training/horovod_adapters.h"
#include "horovod_kernels.h"

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    HorovodAllReduceOp,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    HorovodAllReduceOp);

Status HorovodAllReduceOp::Compute(OpKernelContext* context) const {
   
   ORT_RETURN_IF_ERROR(ConvertStatus(horovod::common::CheckInitialized()));

   const Tensor* input_tensor = context->Input<Tensor>(0);
   auto device_id = context->GetDeviceId();
   auto hvd_input = std::make_shared<ORTTensor>(input_tensor);
   auto hvd_context = std::make_shared<ORTOpContext>(context);
   auto hvd_output = std::make_shared<ORTTensor>(context->Output(0, input_tensor->Shape()));

    bool ready = false;

   ORT_RETURN_IF_ERROR (
      ConvertStatus(
         EnqueueTensorAllreduce(
         hvd_context, hvd_input, hvd_output, 0/*ready_event*/,
         unique_name, device_id,
         [&ready, this](const horovod::common::Status&) {
            {
               std::unique_lock<std::mutex> lck(mutex_);
               ready = true;
            }
            cv_.notify_one();
         })
      )
   );

    {
       std::unique_lock<std::mutex> lk(mutex_);
       cv_.wait(lk,
       [&ready](){
             return ready;
          });
    }
    return Status::OK();
}
}
