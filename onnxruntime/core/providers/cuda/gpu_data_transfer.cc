// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/gpu_data_transfer.h"
#include "cuda_common.h"

#include <iomanip>
#include <iostream>
#include <fstream>

namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer() {
  // create streams, default is nullptr
  streams_[kCudaStreamDefault] = nullptr;
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyIn], cudaStreamNonBlocking));
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyOut], cudaStreamNonBlocking));
}

GPUDataTransfer::~GPUDataTransfer() {
  CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyIn]));
  CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyOut]));
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::CUDA_PINNED
         || dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::CUDA_PINNED;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU && src_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, streams_[exec_queue_id]));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, streams_[kCudaStreamDefault]));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU && dst_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, streams_[exec_queue_id]));
    } else {
      // copying from GPU to CPU memory, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

void PrintTensor(const Tensor& tensor, const std::string& name, bool summary)
{
  if (strcmp(tensor.Location().name, CUDA) != 0)
    return;

  const PrimitiveDataTypeBase* pdt = dynamic_cast<const PrimitiveDataTypeBase*>(tensor.DataType());
  if (!pdt)
    return;

  if (pdt->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* tensor_data = tensor.template Data<float>();
    const TensorShape shape{tensor.Shape()};
    std::vector<float> tensor_cpu(shape.Size());
    cudaMemcpy(&tensor_cpu[0], tensor_data, sizeof(float) * tensor_cpu.size(), cudaMemcpyDeviceToHost);
    if (summary)
    {
      float sum = 0, sum_sq = 0;
      for (size_t i = 0; i < tensor_cpu.size(); i++) {
        sum += tensor_cpu[i];
        sum_sq += tensor_cpu[i] * tensor_cpu[i];
      }
      std::cout << name << ": " << std::setprecision(std::numeric_limits<float>::digits10 + 3) << sum << " sq: "<< sum_sq << std::endl;
    } else {
      // save the data
      std::string filename = name;
      std::replace(filename.begin(), filename.end(), ':', '-');
      std::replace(filename.begin(), filename.end(), ' ', '-');
      std::replace(filename.begin(), filename.end(), '/', '-');

      std::ofstream wf(std::string("/bert_ort/liqun/test_out/") + filename + ".dat", std::ios::out | std::ios::binary);
      wf.write((char *)(&tensor_cpu[0]), tensor_cpu.size() * sizeof(float));
      wf.close();

      std::cout << name << ": " << std::endl;
      const TensorShape shape{tensor.Shape()};
      if (tensor_cpu.size() <= 64 || shape.NumDimensions() == 1) {
        for (size_t i = 0; i < tensor_cpu.size(); i++) {
          std::cout << std::setprecision(std::numeric_limits<float>::digits10 + 3) << tensor_cpu[i] << std::endl;
        }
      } else {
        int dims = shape.NumDimensions();
        int stride = shape[dims - 1];
        int total = 1;
        for (size_t dim = 0; dim < shape.NumDimensions() - 1; dim++) {
          total *= shape[dim];
        }
        for (int count = 0; count < total; count++) {
          float sum = 0;
          for (int i = 0; i < stride; i++) {
            sum += tensor_cpu[count * stride + i];
          }
          std::cout << name << ": " << std::setprecision(std::numeric_limits<float>::digits10 + 3) << sum << std::endl;
        }
      }
    }
  } else if (pdt->GetDataType() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    const int64_t* tensor_data = tensor.template Data<int64_t>();
    const TensorShape shape{tensor.Shape()};
    std::vector<int64_t> tensor_cpu(shape.Size());
    cudaMemcpy(&tensor_cpu[0], tensor_data, sizeof(int64_t) * tensor_cpu.size(), cudaMemcpyDeviceToHost);
    if (summary)
    {
      int64_t sum = 0;
      for (size_t i = 0; i < tensor_cpu.size(); i++)
        sum += tensor_cpu[i];
      std::cout << name << ": " << std::setprecision(std::numeric_limits<int64_t>::digits10 + 1) << sum << std::endl;
    } else {
      std::cout << name << ": " << std::endl;
      for (size_t i = 0; i < tensor_cpu.size(); i++) {
        std::cout << std::setprecision(std::numeric_limits<int64_t>::digits10 + 1) << tensor_cpu[i] << std::endl;
      }
    }
  }
}
}  // namespace onnxruntime
