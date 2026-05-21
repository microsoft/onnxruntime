// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <type_traits>
#include <algorithm>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/neutron/neutron_fwd.h"
#include "core/providers/neutron/neutron_kernel.h"

namespace onnxruntime {
namespace neutron {

namespace NeutronIndex {
// Input index
constexpr int32_t MICROCODE = 0;
constexpr int32_t WEIGHTS = 1;
constexpr int32_t KERNELS = 2;
constexpr int32_t INPUTS = 3;

// Output index
constexpr int32_t SCRATCH = 0;
constexpr int32_t PROFILE = 1;
constexpr int32_t DEBUG = 2;
constexpr int32_t OUTPUTS = 3;
};  // namespace NeutronIndex

class NeutronGraphKernel final : public OpKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NeutronGraphKernel);

  NeutronGraphKernel(const OpKernelInfo& info)
      : OpKernel(info),
        input_count_{info.GetInputCount()},
        output_count_{info.GetOutputCount()},
        nmh_{NULL} {
    // Allocate arrays for inputs and outputs
    dcfg_.inputs = new const void*[input_count_ - NeutronIndex::INPUTS];
    dcfg_.outputs = new void*[output_count_ - NeutronIndex::OUTPUTS];

    for (uint32_t i = 0; i < output_count_; ++i) {
      auto shape_proto = info.node().OutputDefs().at(i)->Shape();
      auto shape = utils::GetTensorShapeFromTensorShapeProto(*shape_proto);
      output_shapes_.push_back(shape);
    }
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override {
    ORT_UNUSED_PARAMETER(alloc);
    ORT_UNUSED_PARAMETER(is_packed);
    ORT_UNUSED_PARAMETER(prepacked_weights);
    switch (input_idx) {
      case NeutronIndex::MICROCODE: {
        mcfg_.microcode = tensor.DataRaw();
        break;
      }
      case NeutronIndex::WEIGHTS: {
        mcfg_.weights = tensor.DataRaw();
        break;
      }
      case NeutronIndex::KERNELS: {
        mcfg_.kernels = tensor.DataRaw();
        auto ret = neutronModelPrepare(&mcfg_, &nmh_);
        ORT_ENFORCE(ret == ENONE, "NeutronGraph model prepare error");
        break;
      }
    }
    return Status::OK();
  }

  Status Compute(OpKernelContext* ctx) const override {
    uint32_t i;
    // Set reference for all inputs
    for (i = NeutronIndex::INPUTS; i < input_count_; i++) {
      const auto* tensor = ctx->Input<Tensor>(i);
      dcfg_.inputs[i - NeutronIndex::INPUTS] = tensor->DataRaw();
    }

    // Set reference for all outputs
    for (i = NeutronIndex::OUTPUTS; i < output_count_; ++i) {
      auto* output = ctx->Output(i, output_shapes_[i]);
      dcfg_.outputs[i - NeutronIndex::OUTPUTS] = output->MutableDataRaw();
    }
    // Run neutron compute.
    auto ret = neutronRunBlocking(nmh_, &dcfg_);
    if (ret != ENONE) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "NeutronEP:NeutronGraph falied to invoke");
    }

    return Status::OK();
  }

  ~NeutronGraphKernel() override {
    neutronModelUnprepare(nmh_);
    delete dcfg_.inputs;
    delete dcfg_.outputs;
  }

 private:
  const uint32_t input_count_;
  const uint32_t output_count_;
  std::vector<TensorShape> output_shapes_;

  NeutronModelConfig mcfg_;
  NeutronDataConfig dcfg_;
  NeutronModelHandle nmh_;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    NeutronGraph,
    kNeutronDomain,
    1,
    int8_t,
    kNeutronExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>()),
    onnxruntime::neutron::NeutronGraphKernel);
}  // namespace neutron
}  // namespace onnxruntime
