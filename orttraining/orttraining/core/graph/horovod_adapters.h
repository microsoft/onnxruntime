// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/framework/op_kernel.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#endif

#include "operations.h"

namespace onnxruntime {

namespace hvd = horovod::common;

common::Status ConvertStatus(const hvd::Status& status);
hvd::Status ConvertStatus(const common::Status& status);
const hvd::ReduceOp GetReduceOp(const int64_t reduce_op_enum);

class ORTTensor : public hvd::Tensor {

public:
  ORTTensor(const onnxruntime::Tensor* tensor);
  virtual const hvd::DataType dtype() const override;
  virtual const hvd::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

private:
  const onnxruntime::Tensor* tensor_;
};

class ORTPersistentBuffer : public hvd::PersistentBuffer {
 public:
  ORTPersistentBuffer(AllocatorPtr allocator, int64_t size);
  virtual ~ORTPersistentBuffer();

  virtual const void* AccessData(std::shared_ptr<hvd::OpContext> context) const override;

 private:
  AllocatorPtr allocator_;
  void* buffer_ = nullptr;
};

class ORTOpContext : public hvd::OpContext {
public:
  ORTOpContext(AllocatorPtr allocator);

  virtual hvd::Status AllocatePersistent(int64_t size, std::shared_ptr<hvd::PersistentBuffer>* tensor) override;

  virtual hvd::Status AllocateOutput(hvd::TensorShape shape, std::shared_ptr<hvd::Tensor>* tensor) override;

  virtual hvd::Framework framework() const override;

  virtual hvd::Status AllocateZeros(int64_t num_elements, hvd::DataType dtype, std::shared_ptr<hvd::Tensor>* tensor) override;

private:
  AllocatorPtr allocator_;
};

}  // namespace onnxruntime

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif