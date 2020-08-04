// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include <cmath>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/providers/acl/nn/pool.h"
#include "core/providers/acl/acl_common.h"
#include "core/providers/acl/acl_fwd.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

#define PREF_DIM 4

namespace onnxruntime {
namespace acl {

template <typename T, typename PoolType>
thread_local std::map<OpKernel*, ACLNEPool> Pool<T, PoolType>::poolLayers;

template <typename T>
thread_local std::map<OpKernel*, ACLNEPool> MaxPoolV8<T>::maxPoolLayers;

template <typename T>
ACLNEPool PoolOperation(onnxruntime::OpKernelContext* context,
                     arm_compute::PoolingType pool_type,
                     onnxruntime::PoolAttributes pool_attrs,
                     PoolLayersIterator it,
                     bool insert){

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  std::vector<int64_t> pads = pool_attrs.pads;
  std::vector<int64_t> strides = pool_attrs.strides;
  std::vector<int64_t> kernel_shape = pool_attrs.kernel_shape;

  if (pool_attrs.global_pooling) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = pool_attrs.SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, TensorShape(output_dims));

  ACLNEPool tpool;
  if (insert) {
    auto layer = std::make_shared<arm_compute::NEPoolingLayer>();

    tpool.in = std::make_shared<arm_compute::Tensor>();
    tpool.out = std::make_shared<arm_compute::Tensor>();

    tpool.in->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(X->Shape(), PREF_DIM), arm_compute::Format::F32));
    tpool.out->allocator()->init(arm_compute::TensorInfo(ACLTensorShape(Y->Shape(), PREF_DIM), arm_compute::Format::F32));

    if (pool_attrs.global_pooling) {
      layer->configure(tpool.in.get(), tpool.out.get(), arm_compute::PoolingLayerInfo(pool_type));
    } else {
      std::vector<int64_t> aclStrides(2);
      aclStrides[0] = (strides.size() == 2) ? strides[1] : 1;
      aclStrides[1] = strides[0];

      std::vector<int64_t> aclPads(4);
    // The pad order in acl is: pad_left, pad_right, pad_top, pad_bottom
      if (pads.size() == 2) {
        if (strides.size() == 1) {
          aclPads[0] = 0;
          aclPads[1] = 0;
          aclPads[2] = pads[1];
          aclPads[3] = pads[0];
        } else {
          aclPads[0] = pads[1];
          aclPads[1] = pads[0];
          aclPads[2] = pads[1];
          aclPads[3] = pads[0];
        }
      } else {
        aclPads[0] = pads[1];
        aclPads[1] = pads[3];
        aclPads[2] = pads[0];
        aclPads[3] = pads[2];
      }

      arm_compute::PadStrideInfo aclPadStride = arm_compute::PadStrideInfo(aclStrides[0], aclStrides[1],
                                                                           aclPads[0], aclPads[1], aclPads[2], aclPads[3], arm_compute::DimensionRoundingType::FLOOR);

      std::vector<int64_t> aclKernelShape(2);
      aclKernelShape[0] = (kernel_shape.size() > 1) ? kernel_shape[1] : 1;
      aclKernelShape[1] = kernel_shape[0];

      arm_compute::Size2D aclSize(aclKernelShape[0], aclKernelShape[1]);

      bool excludePadding = (pool_type == arm_compute::PoolingType::AVG && pool_attrs.count_include_pad) ? false : true;

      LOGS_DEFAULT(VERBOSE) << "padding: {" << aclPads[0] << "," << aclPads[1] << "," << aclPads[2] << "," << aclPads[3] << "}";
      LOGS_DEFAULT(VERBOSE) << "kernel shape: {" << aclKernelShape[0] << "," << aclKernelShape[1] << "}";
      LOGS_DEFAULT(VERBOSE) << "strides: {" << aclStrides[0] << "," << aclStrides[1] << "}";
      LOGS_DEFAULT(VERBOSE) << "excludePadding: " << excludePadding;

      arm_compute::PoolingLayerInfo pool_info(pool_type, aclSize, aclPadStride, excludePadding);
      layer->configure(tpool.in.get(), tpool.out.get(), pool_info);
    }

    // allocate space for input tensor to accomodate paddings and strides
    tpool.in->allocator()->allocate();

    tpool.layer = std::move(layer);
  } else {
    tpool = it->second;
  }
  const T* x_data = X->template Data<T>();
  arm_compute::Window aclInpuWindow;
  aclInpuWindow.use_tensor_dimensions(tpool.in->info()->tensor_shape());

  arm_compute::Iterator aclInputIt(tpool.in.get(), aclInpuWindow);
  const unsigned int aclWidth = tpool.in->info()->dimension(0);
  const unsigned int aclHeight = tpool.in->info()->dimension(1);

  // copy input tensor into the larger buffer
  arm_compute::execute_window_loop(
      aclInpuWindow,
      [&](const arm_compute::Coordinates& co) {
        *reinterpret_cast<float*>(aclInputIt.ptr()) = x_data[co.z() * (aclWidth * aclHeight) + co.y() * aclHeight + co.x()];
      },
      aclInputIt);

  T* y_data = Y->template MutableData<T>();
  ACLImportMemory(tpool.out->allocator(), (void*)y_data, Y->Shape().Size() * 4);

  tpool.layer->run();

  return tpool;
}

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {
  arm_compute::Tensor in, out;

  const Tensor* X = context->Input<Tensor>(0);

  std::vector<int64_t> dilations(PoolBase::pool_attrs_.dilations);
  std::vector<int64_t> aclDilations(2);
  aclDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
  aclDilations[1] = (!dilations.empty()) ? dilations[0] : 1;

  if (X->Shape().NumDimensions() != PREF_DIM) {
    LOGS_DEFAULT(WARNING) << "ArmNN does not have support for tensors with 4 or more dimensions; defaulting to cpu implementation";
    Status s = onnxruntime::Pool<T, PoolType>::Compute(context);
    return s;
  }

  if (aclDilations[0] * aclDilations[1] > 1) {
    LOGS_DEFAULT(WARNING) << "ArmNN does not have support for dilation; defaulting to cpu implementation";
    Status s = onnxruntime::Pool<T, PoolType>::Compute(context);
    return s;
  }

  arm_compute::PoolingType pool_type;
  if (PoolBase::op_name_ == "GlobalAveragePool" || PoolBase::op_name_ == "AveragePool")
    pool_type = arm_compute::PoolingType::AVG;
    LOGS_DEFAULT(VERBOSE) << "AveragePool";
  else if (PoolBase::op_name_ == "GlobalMaxPool" || PoolBase::op_name_ == "MaxPool")
    pool_type = arm_compute::PoolingType::MAX;
    LOGS_DEFAULT(VERBOSE) << "MaxPool";
  else {
    LOGS_DEFAULT(WARNING) << "Pooling operation not supported in ArmNN; defaulting to cpu implementation";
    return onnxruntime::Pool<T, PoolType>::Compute(context);
  }

  PoolLayersIterator it = Pool::poolLayers.find((OpKernel*) this);
  bool insert = it == Pool::poolLayers.end();
  ACLNEPool pPool = PoolOperation<T>(context, pool_type, PoolBase::pool_attrs_, it, insert);
  if(insert){
    std::pair<PoolLayersIterator, bool> ret;
    ret = Pool::poolLayers.insert(std::pair<OpKernel*, ACLNEPool>((OpKernel*) this, pPool));
  }

  LOGS_DEFAULT(VERBOSE) << std::endl;

  return Status::OK();
}

template <typename T>
Status MaxPoolV8<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);

  std::vector<int64_t> dilations(PoolBase::pool_attrs_.dilations);
  std::vector<int64_t> aclDilations(2);
  aclDilations[0] = (dilations.size() == 2) ? dilations[1] : 1;
  aclDilations[1] = (!dilations.empty()) ? dilations[0] : 1;

  if (X->Shape().NumDimensions() != PREF_DIM) {
    LOGS_DEFAULT(WARNING) << "ArmNN does not have support for tensors with 4 or more dimensions; defaulting to cpu implementation";
    Status s = onnxruntime::MaxPoolV8::Compute(context);
    return s;
  }

  if (aclDilations[0] * aclDilations[1] > 1) {
    LOGS_DEFAULT(WARNING) << "ArmNN does not have support for dilation; defaulting to cpu implementation";
    Status s = onnxruntime::MaxPoolV8::Compute(context);
    return s;
  }

  LOGS_DEFAULT(VERBOSE) << "MaxPoolV8";

  PoolLayersIterator it = MaxPoolV8::maxPoolLayers.find((OpKernel*) this);
  bool insert = it == MaxPoolV8::maxPoolLayers.end();
  ACLNEPool pPool = PoolOperation<T>(context, arm_compute::PoolingType::MAX, PoolBase::pool_attrs_, it, insert);
  if(insert){
    std::pair<PoolLayersIterator, bool> ret;
    ret = MaxPoolV8::maxPoolLayers.insert(std::pair<OpKernel*, ACLNEPool>((OpKernel*) this, pPool));
  }

  LOGS_DEFAULT(VERBOSE) << std::endl;

  return Status::OK();
}

#define POOLING_KERNEL(op_name, data_type, pool_type, since_version, end_version)       \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                              \
      op_name,                                                                          \
      kOnnxDomain,                                                                      \
      since_version,                                                                    \
      end_version,                                                                      \
      data_type,                                                                        \
      kAclExecutionProvider,                                                            \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Pool<data_type, pool_type>);

POOLING_KERNEL(MaxPool, float, MaxPool<1>, 1, 7)
POOLING_KERNEL(AveragePool, float, AveragePool, 7, 9)
POOLING_KERNEL(AveragePool, float, AveragePool, 10, 10)
POOLING_KERNEL(GlobalAveragePool, float, AveragePool, 1, 8)
POOLING_KERNEL(GlobalMaxPool, float, MaxPool<1>, 1, 8)

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                            \
      MaxPool,                                                                      \
      kOnnxDomain,                                                                  \
      8,                                                                            \
      11,                                                                           \
      float,                                                                        \
      kAclExecutionProvider,                                                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      MaxPoolV8<float>);

}  // namespace acl
}  // namespace onnxruntime