1. how opkernel context know node's input/output tensor id?
2. how multiple stream is sync?
   1. sync by queue_id
   2. CUDAFence里的sync好像都是data_transfer的stream?
      1. https://github.com/microsoft/onnxruntime/blob/c99aa3a3f3834adcbb888ce4b964f2695c524eae/onnxruntime/core/providers/cuda/cuda_fence.cc#L23
      2.
3. how onnx node executed?
   1. Node >> NodeIndex
   2. OpKernel = SessionState.session_kernels_[NodeIndex]
   3. OpKernel.compute(OpKernelContext)
4. how OpKernel.compute get its input/output OrtValue?
   1. node_input_start_index_ = IExecutionFrame->GetNodeOffset(kernel->Node().Index())
   2. node_output_start_index_ = node_input_start_index_ + InputCount + ImplicitInputCount
   3. NodeIndex = node_output_start_index_ + ith_output
   4. ort_value_idx = NodeIndexInfo[NodeIndex]
   5. OrtValue = all_values_[ort_value_idx]
   6. OrtValue.IsAllocated == False, 则AllocateAsPerAllocationPlan > 根据AllocKind去看怎么拿到, e.g. reuse/alloc
      1. reuse时, AllocateMLValueTensorPreAllocateBuffer >> 拿到被reuse的OrtValue的tenson去create new tensor
5. how OrtValue being released?
6. how memory allocation plan is made? is executed?
7. NodeArg vs NodeDef
   1. n:1的对应关系
   2. NodeDef指的是ONNX协议下, node的第ith个input/output, 此时的一个input可以对应多个NodeArg, e.g. concat的第一个input, 可以对应多个input, 也即多个edge
   3. nodeArg就是onnx graph上的一条edge
8.
