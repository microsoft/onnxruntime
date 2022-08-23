// AUTO-GENERATED CODE! - DO NOT EDIT!
// $ python /bert_ort/chenta/onnxruntime/orttraining/orttraining/eager/opgen/opgen.py --output_file /bert_ort/chenta/onnxruntime/orttraining/orttraining/eager/ort_aten.g.cpp.working --ops_module /bert_ort/chenta/onnxruntime/orttraining/orttraining/eager/opgen/opgen/atenops.py --header_file /bert_ort/chenta/eager_ort/lib/python3.7/site-packages/torch/include/ATen/RegistrationDeclarations.h

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "my_tensor.h"
#include "my_allocator.h"
#include "my_kernel.h"

namespace torch_my_kernel_lib {

using namespace at;
namespace aten {

at::Tensor empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,  // Ignored because there's no ONNX support.
    c10::optional<at::Device> device_opt,  // Will be ORT by the time this is dispatched.
    c10::optional<bool> pin_memory_opt) {  // Ignored because there's no ONNX support.
  assert(device_opt.has_value());
  at::ScalarType dtype = c10::dtype_or_default(dtype_opt);
  if (dtype != at::ScalarType::Float) {
    throw std::runtime_error("Unsupported dtype");
  }
  // calculate num of elements from size
  int64_t num_elements = 1;
  for (size_t i = 0; i < size.size(); ++i) {
    num_elements *= size[i];
  }

  void* data = my_kernel_lib::my_alloc(num_elements * sizeof(float));

  delete_function delete_fn = [](void* data) {
    if (data) {
      my_kernel_lib::my_free(data);
    }
  };

  std::vector<int64_t> shape(size.begin(), size.end());

  std::shared_ptr<MyTensor> tensor = std::make_shared<MyTensor>(data, shape, delete_fn);

  return at::Tensor(c10::make_intrusive<MyATenTensorImpl>(
      std::move(tensor),
      at::TensorOptions()
          .device(*device_opt)
          .dtype(dtype)));
}

at::Tensor empty_memory_format(
    at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt,
    c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {  // Ignored because there's no ONNX support.

  return empty_strided(size, at::IntArrayRef({}), dtype_opt, layout_opt, device_opt, pin_memory);
}

// aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor& abs_out(
    const Tensor& self,
    // *,
    Tensor& out) {
  throw std::runtime_error("not implemented");
}

at::Tensor& copy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  auto device_self = self.device();
  auto device_src = src.device();

  if (self.scalar_type() != src.scalar_type()) {
    throw std::runtime_error("copy: src and self must have the same dtype");
  }

  if (self.scalar_type() != at::ScalarType::Float) {
    throw std::runtime_error("copy: only float is supported");
  }

  if ((device_self.type() != DeviceType::ORT && device_self.type() != DeviceType::CPU) ||
      (device_src.type() != DeviceType::ORT && device_src.type() != DeviceType::CPU)) {
    throw std::runtime_error("copy_ with non ORT/CPU device is not supported");
  }

  if (device_self.type() == DeviceType::CPU && device_src.type() == DeviceType::CPU) {
    auto self_data = self.data_ptr<float>();
    auto src_data = src.data_ptr<float>();
    memcpy(self_data, src_data, src.numel() * sizeof(float));
  } else if (device_self.type() == DeviceType::CPU && device_src.type() == DeviceType::ORT) {
    auto* impl = dynamic_cast<MyATenTensorImpl*>(src.unsafeGetTensorImpl());
    if (!impl) {
      throw std::runtime_error("unexpected");
    }
    auto& tensor = impl->tensor();
    auto* self_data = self.data_ptr<float>();
    auto* src_data = tensor.buffer();
    memcpy(self_data, src_data, self.numel() * sizeof(float));
  } else if (device_self.type() == DeviceType::ORT && device_src.type() == DeviceType::CPU) {
    auto* impl = dynamic_cast<MyATenTensorImpl*>(self.unsafeGetTensorImpl());
    if (!impl) {
      throw std::runtime_error("unexpected");
    }
    auto& tensor = impl->tensor();
    auto* src_data = src.data_ptr<float>();
    auto* self_data = tensor.buffer();
    memcpy(self_data, src_data, self.numel() * sizeof(float));
  } else if (device_self.type() == DeviceType::ORT && device_src.type() == DeviceType::ORT) {
    auto* impl = dynamic_cast<MyATenTensorImpl*>(self.unsafeGetTensorImpl());
    if (!impl) {
      throw std::runtime_error("unexpected");
    }
    auto& self_tensor = impl->tensor();
    auto* self_data = self_tensor.buffer();
    auto* src_impl = dynamic_cast<MyATenTensorImpl*>(src.unsafeGetTensorImpl());
    if (src_impl) {
      throw std::runtime_error("unexpected");
    }
    auto& src_tensor = src_impl->tensor();
    auto* src_data = src_tensor.buffer();
    memcpy(self_data, src_data, self.numel() * sizeof(float));
  } else {
    throw std::runtime_error("unexpected");
  }
  return self;
}

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor& add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    // *,
    const at::Scalar& alpha,
    at::Tensor& out) {
  if (self.scalar_type() != other.scalar_type()) {
    throw std::runtime_error("add: src and self must have the same dtype");
  }
  if (self.scalar_type() != at::ScalarType::Float) {
    throw std::runtime_error("add: only float is supported");
  }

  auto* self_impl = dynamic_cast<MyATenTensorImpl*>(self.unsafeGetTensorImpl());
  if (!self_impl) {
    throw std::runtime_error("unexpected");
  }

  auto& self_tensor = self_impl->tensor();

  auto* other_impl = dynamic_cast<MyATenTensorImpl*>(other.unsafeGetTensorImpl());
  if (!other_impl) {
    throw std::runtime_error("unexpected");
  }
  auto& other_tensor = other_impl->tensor();

  // calculate output shape and type
  auto& self_shape = self_tensor.sizes();
  auto& other_shape = other_tensor.sizes();
  my_kernel_lib::DataType self_type = my_kernel_lib::DataType::kFloat;
  my_kernel_lib::DataType other_type = my_kernel_lib::DataType::kFloat;
  my_kernel_lib::DataType output_type;
  std::vector<int64_t> output_shape;
  auto status = AddKernelTypeShapeInference(self_type, self_shape,
                                            other_type, other_shape,
                                            &output_type, &output_shape);
  if (status != my_kernel_lib::Status::kOK) {
    throw std::runtime_error("add: shape inference failed");
  }

  // calculate output buffer size
  int64_t output_size = 1;
  for (int i = 0; i < output_shape.size(); i++) {
    output_size *= output_shape[i];
  }

  // create output tensor
  void* buffer = my_kernel_lib::my_alloc(output_size * sizeof(float));

  // invoke add
  status = my_kernel_lib::AddKernel<float>(static_cast<float*>(self_tensor.buffer()), self_shape,
                                           static_cast<float*>(other_tensor.buffer()), other_shape,
                                           static_cast<float*>(buffer), output_shape);
  if (status != my_kernel_lib::Status::kOK) {
    throw std::runtime_error("add: kernel failed");
  }

  auto* out_impl = dynamic_cast<MyATenTensorImpl*>(out.unsafeGetTensorImpl());
  if (!out_impl) {
    throw std::runtime_error("unexpected");
  }
  auto* out_tensor = out_impl->mutable_tensor();
  out_tensor->resize(output_shape, buffer);
  return out;
}

}  // namespace aten

TORCH_LIBRARY_IMPL(aten, ORT, m) {
  m.impl("aten::empty.memory_format", TORCH_FN(aten::empty_memory_format));
  m.impl("aten::empty_strided", TORCH_FN(aten::empty_strided));
  m.impl("aten::copy_", TORCH_FN(aten::copy_));
  m.impl("aten::add.out", TORCH_FN(aten::add_out));
}

PYBIND11_MODULE(my_kernel_aten_backend, m) {
  m.doc() = "pybind11 for my_kernel";
}

}  // namespace torch_my_kernel_lib
