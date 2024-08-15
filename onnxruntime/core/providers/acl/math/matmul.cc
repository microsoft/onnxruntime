// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <memory>
#include "core/common/status.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor_shape.h"
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
#include <thread>
#include <mutex>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/acl/math/matmul.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"

// ACL
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/functions/NEMatMul.h"
#include "src/cpu/operators/CpuGemm.h"
#include "src/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"
#include "src/cpu/operators/CpuMatMul.h"


namespace onnxruntime {

namespace acl {

TensorShape BroadcastInput(const TensorShape &shape, bool prependDim) {
  const auto nd = shape.NumDimensions();
  if (nd == 0) {
    ORT_THROW("MatMul by scalar not allowed");
  }

  int64_t batchSize = 1;
  if (nd == 1) {
    if (prependDim) {
      return {1, 1, shape[0]};
    } else {
      return {1, shape[0], 1};
    }
  }

  for (size_t i = 0; i < nd - 2; i++) {
      batchSize *= shape[i];
  }

  return {batchSize, shape[nd - 2], shape[nd - 1]};
}

struct MatMulConfig {
  bool isQuantized;
  float alpha;
  bool transA;
  bool transB;
  TensorShape aShapeBroadcast;
  TensorShape bShapeBroadcast;
};

Status ParseMatMul(const onnxruntime::Node& node, MatMulConfig &config) {
  onnxruntime::ProtoHelperNodeContext ctx(node);
  onnxruntime::OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
  const auto inputDefs = node.InputDefs();

  config.isQuantized = node.OpType() == "MatMulIntegerToFloat";

  config.alpha = 1;
  attrs.GetAttr("alpha", &config.alpha);

  int64_t transA = 0;
  attrs.GetAttr("transA", &transA);
  int64_t transB = 0;
  attrs.GetAttr("transB", &transB);

  config.transA = transA;
  config.transB = transB;

  const int64_t transBatchA = attrs.GetAttrOrDefault<int64_t>("transBatchA", 0);
  const int64_t transBatchB = attrs.GetAttrOrDefault<int64_t>("transBatchB", 0);

  ORT_RETURN_IF(transBatchA, "transBatchA not supported by ACL");
  ORT_RETURN_IF(transBatchB, "transBatchB not supported by ACL");

  ORT_RETURN_IF(config.isQuantized && inputDefs.size() >= 7, "ACL MatMulIntegerToFloat does not support bias");

  TensorShape aShapeIn;
  ORT_RETURN_IF_ERROR(GetArgShape(inputDefs[0], aShapeIn));

  TensorShape bShapeIn;
  ORT_RETURN_IF_ERROR(GetArgShape(inputDefs[1], bShapeIn));

  config.aShapeBroadcast = BroadcastInput(aShapeIn, !config.transA);
  config.bShapeBroadcast = BroadcastInput(bShapeIn, config.transB);

  ORT_RETURN_IF(!(config.bShapeBroadcast[0] == 1 || (config.aShapeBroadcast[0] == config.bShapeBroadcast[0])),
      "ACL does not support broadcasting");

  ORT_RETURN_IF(config.alpha != 1 && config.bShapeBroadcast[0] > 1,
      "ACL does not support alpha scaling with batched B");

  return Status::OK();
}

Status ValidateMatMul(const onnxruntime::Node& node) {
  MatMulConfig config;
  return ParseMatMul(node, config);
}

MatMul::MatMul(const OpKernelInfo& info): onnxruntime::OpKernel(info) {
  provider_ = (const_cast<ACLExecutionProvider*>(
      static_cast<const ACLExecutionProvider*>(info.GetExecutionProvider())));

  const auto inputDefs = OpKernel::Node().InputDefs();
  const auto outputDefs = OpKernel::Node().OutputDefs();

  const Tensor *tmp = nullptr;
  const bool aIsConst = info.TryGetConstantInput(0, &tmp);
  const bool bIsConst = info.TryGetConstantInput(1, &tmp);

  MatMulConfig config;
  ORT_THROW_IF_ERROR(ParseMatMul(OpKernel::Node(), config));

  ORT_THROW_IF_ERROR(GetArgShape(outputDefs[0], outShape));
  if (outShape.Size() == 0) {
    return;
  }

  const TensorShape aShape {
    config.aShapeBroadcast[0],
    config.aShapeBroadcast[config.transA? 2 : 1],
    config.aShapeBroadcast[config.transA? 1 : 2]
  };

  const TensorShape bShape {
    config.bShapeBroadcast[0],
    config.bShapeBroadcast[config.transB? 2 : 1],
    config.bShapeBroadcast[config.transB? 1 : 2]
  };

  const TensorShape outShapeBroadcast {aShape[0], aShape[1], bShape[2]};

  ORT_ENFORCE(outShape.Size() == outShapeBroadcast.Size(), "Output sizes do not match");

  arm_compute::DataType aType = ACLDataType(*inputDefs[0]->Type());
  arm_compute::DataType bType = ACLDataType(*inputDefs[1]->Type());
  arm_compute::DataType outType = ACLDataType(*outputDefs[0]->Type());

  arm_compute::GEMMInfo gemmInfo(false, false, bIsConst);
  gemmInfo.set_fast_math(provider_->info.enable_fast_math);

  a = std::make_shared<arm_compute::Tensor>();
  b = std::make_shared<arm_compute::Tensor>();
  out = std::make_shared<arm_compute::Tensor>();

  a->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(config.aShapeBroadcast), 1, aType));
  b->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(config.bShapeBroadcast), 1, bType));
  out->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(outShapeBroadcast), 1, outType));

  if (config.isQuantized) {
    ORT_THROW_IF_ERROR(LoadQuantizationInfo(info, a.get(), 2, 4, true));
    ORT_THROW_IF_ERROR(LoadQuantizationInfo(info, b.get(), 3, 5, true));
  }

  arm_compute::ITensor *a_to_use = a.get();
  if (config.transA) {
    a_transposed = std::make_shared<arm_compute::Tensor>();
    a_transposed->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(aShape), 1, aType));
    a_to_use = a_transposed.get();

    a_permute = std::make_shared<arm_compute::NEPermute>();
    a_permute->configure(a.get(), a_transposed.get(), {1, 0, 2});
  }

  arm_compute::ITensor *b_to_use = b.get();
  if (config.transB) {
    if (bIsConst) {
      workspace.persistent_tensors.emplace_back(std::make_unique<arm_compute::Tensor>());
      b_transposed = workspace.persistent_tensors.back().get();
    } else {
      workspace.temporary_tensors.emplace_back(std::make_unique<arm_compute::Tensor>());
      b_transposed = workspace.temporary_tensors.back().get();
    }

    b_transposed->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(bShape), 1, bType), 128);
    b_to_use = b_transposed;

    b_permute = std::make_shared<arm_compute::NEPermute>();
    b_permute->configure(b.get(), b_transposed, {1, 0, 2});
  }

  a_to_use->info()->set_are_values_constant(aIsConst);
  b_to_use->info()->set_are_values_constant(bIsConst);

  if (config.bShapeBroadcast[0] > 1) {
    arm_compute::CpuMatMulSettings settings;
    settings.fast_math(provider_->info.enable_fast_math);

    a_to_use->info()->set_are_values_constant(false);
    b_to_use->info()->set_are_values_constant(false);

    const auto matmul = std::make_shared<arm_compute::cpu::CpuMatMul>();
    matmul->configure(a_to_use->info(), b_to_use->info(), out->info(), {}, settings, {});
    layer = std::move(matmul);
  } else if (config.isQuantized) {
    const auto gemm = std::make_shared<arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore>();
    gemm->configure(a_to_use->info(), b_to_use->info(), nullptr, out->info(), gemmInfo);
    layer = std::move(gemm);
  } else {
    const auto gemm = std::make_shared<arm_compute::cpu::CpuGemm>();
    gemm->configure(a_to_use->info(), b_to_use->info(), nullptr, out->info(), config.alpha, 0.f, gemmInfo);
    layer = std::move(gemm);
  }

  memory_group = arm_compute::MemoryGroup(provider_->memory_manager);
  run_pack = {{arm_compute::ACL_SRC_0, a_to_use}, {arm_compute::ACL_SRC_1, b_to_use},
              {arm_compute::ACL_DST, out.get()}};
  prep_pack = {{arm_compute::ACL_SRC_1, b_to_use}};

  PopulateWorkspace(layer->workspace(), workspace, memory_group, run_pack, prep_pack);
}

Status MatMul::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
        /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) {

  is_packed = false;
  if (input_idx != 1  || outShape.Size() == 0) {
    return Status::OK();
  }

  const uint8_t *data = (uint8_t *) tensor.DataRaw();

  ORT_RETURN_IF_ERROR(ACLImportMemory(b->allocator(), (void *) data, 0));

  if (!workspace.persistent_tensors.empty()) {
    size_t packedSize = 0;
    size_t alignment = 0;
    GetPackingInfo(workspace.persistent_tensors, packedSize, alignment);
    auto buffSize = packedSize + alignment;

    pbRaw = IAllocator::MakeUniquePtr<void>(alloc, buffSize, true);
    ORT_RETURN_IF_ERROR(LoadPackedTensors(workspace.persistent_tensors, pbRaw.get(), packedSize, alignment));

    if (prepacked_weights != nullptr) {
      prepacked_weights->buffers_.push_back(std::move(pbRaw));
      prepacked_weights->buffer_sizes_.push_back(buffSize);
    }

    is_packed = true;
  }

  if (b_transposed) {
    b_permute->run();
  }

  for (std::unique_ptr<arm_compute::Tensor> &prep_tensor : workspace.prepare_tensors) {
    prep_tensor->allocator()->allocate();
  }

  layer->prepare(prep_pack);

  for (std::unique_ptr<arm_compute::Tensor> &prep_tensor : workspace.prepare_tensors) {
    prep_tensor->allocator()->free();
  }

  return Status::OK();
}

Status MatMul::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                 int input_idx, /*out*/ bool& used_shared_buffers) {

  used_shared_buffers = false;
  if (input_idx != 1) {
    return Status::OK();
  }

  if (!workspace.persistent_tensors.empty()) {
    size_t packedSize = 0;
    size_t alignment = 0;
    GetPackingInfo(workspace.persistent_tensors, packedSize, alignment);

    ORT_RETURN_IF_ERROR(LoadPackedTensors(workspace.persistent_tensors, prepacked_buffers[0].get(), packedSize, alignment));

    used_shared_buffers = true;
  }

  return Status::OK();
}

Status MatMul::Compute(OpKernelContext* context) const {
  provider_->SetThreadPool(context->GetOperatorThreadPool());

  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* B = pbRaw? nullptr : context->Input<Tensor>(1);

  Tensor* outOrt = context->Output(0, outShape);

  if (outShape.Size() == 0) {
    return Status::OK();
  }

  const void* a_data = A->DataRaw();
  const void* b_data = B == nullptr? nullptr : B->DataRaw();
  void* out_data = outOrt->MutableDataRaw();

  ORT_RETURN_IF(A->Shape().Size() != 0 && a->info()->has_padding(), "Padded ACL input tensor not supported");
  ORT_RETURN_IF_ERROR(ACLImportMemory(a->allocator(), (void*)a_data, 0));

  if (b_data != nullptr) {
    ORT_RETURN_IF_ERROR(ACLImportMemory(b->allocator(), (void*)b_data, 0));
  }

  ORT_RETURN_IF(outOrt->Shape().Size() != 0 && out->info()->has_padding(), "Padded ACL output tensor not supported");
  ORT_RETURN_IF_ERROR(ACLImportMemory(out->allocator(), (void*)out_data, 0));

  ORT_RETURN_IF(B != nullptr && workspace.persistent_tensors.size(), "Persistent state requires pre-packing");

  if (a_transposed) {
    a_transposed->allocator()->allocate();
    a_permute->run();
  }

  {
    arm_compute::MemoryGroupResourceScope scope_mg(const_cast<arm_compute::MemoryGroup&>(memory_group));
    if (b_transposed && B) {
      b_permute->run();
    }

    layer->run(const_cast<arm_compute::ITensorPack&>(run_pack));
  }

  a->allocator()->free();
  if (B != nullptr)
    b->allocator()->free();
  out->allocator()->free();

  if (a_transposed) {
    a_transposed->allocator()->free();
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMul,
    kOnnxDomain,
    13,
    MLFloat16,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    MatMul);

ONNX_OPERATOR_KERNEL_EX(
    FusedMatMul,
    kMSDomain,
    1,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedMatMul,
    kMSDomain,
    1,
    MLFloat16,
    kAclExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    MatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    uint8_t,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    int8_t,
    kAclExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMul);

}  // namespace acl
}  // namespace onnxruntime
