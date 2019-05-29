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
    ORTPersistentBuffer(OpKernelContext* context, int64_t size);
    virtual const void* AccessData(std::shared_ptr<hvd::OpContext> context) const override;

private:
     void* buffer_ = nullptr;
};

class ORTOpContext : public hvd::OpContext {
public:
  ORTOpContext(OpKernelContext* context);

  virtual hvd::Status AllocatePersistent(int64_t size, std::shared_ptr<hvd::PersistentBuffer>* tensor) override;

  virtual hvd::Status AllocateOutput(hvd::TensorShape shape, std::shared_ptr<hvd::Tensor>* tensor) override;

  virtual hvd::Framework framework() const override;

private:
  OpKernelContext* context_;
};

}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif