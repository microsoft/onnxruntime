#pragma once
#include "core/providers/brainslice/brainslice_kernel.h"

namespace onnxruntime {
namespace brainslice {

/**
@class BrainSliceOperator
Class representing a hardware accelerated node using the BrainSlice (FPGA) execution provider.

The node schema requires the following attributes to be defined:
- firmware_data:  bytes of the data.bin firmware file (attribute type=String)
- firmware_instructions:  bytes of the instructions.bin firmware file (attribute type=String)
- firmware_schema:  bytes of the schema.bin firmware file (attribute type=String)
- input_addresses:  starting tile address in BrainSlice for the corresponding initializer input
- input_memtypes:  FPGA memory type (matrix/vector, DRAM/on-chip register/etc.) for the corresponding initializer input.  See enum BrainSliceMemList.

The runtime (non-const) inputs must be stored before the const/initializer inputs:
[ runtime_input0, const_input0, const_input1, ..., const_inputN ]

The 'input_addresses' and 'input_memtypes' attribute lengths should match the inputs length.  The address and memtype
for the non-const inputs must be included but can be any value (e.g. -1).

The BrainSlice node can currently have a single non-const input (input0) and a single output (output 0).
*/
class BrainSliceOperator : public BrainSliceOpKernel {
 public:
  explicit BrainSliceOperator(const OpKernelInfo& info);
  virtual ~BrainSliceOperator() {}

  virtual Status Compute(OpKernelContext* context) const override;

 private:
  TensorShape output_dims_;
  bool output_interleaved_ = false;
};

}  // namespace brainslice
}  // namespace onnxruntime
