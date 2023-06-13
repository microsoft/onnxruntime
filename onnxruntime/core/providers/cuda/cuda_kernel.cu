// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"
#include <functional>
#include <iomanip>
#include <iostream>

namespace onnxruntime {
namespace cuda {

#ifdef DEBUG_NODE_INPUTS_OUTPUTS

namespace {

// template <typename T>
// void DumpTensorToStdOut(const Tensor& tensor) {
//   onnxruntime::utils::PrintCpuTensor<T>(tensor, utils::kDefaultSnippetEdgeItems, utils::kDefaultSnippetThreshold);
// }

// void DumpCpuTensor(
//     const Tensor& tensor) {
//   DispatchOnTensorType(tensor.DataType(), DumpTensorToStdOut, tensor);
// }

__global__ void PrintTensorShape(int dims, int dim0, int dim1, int dim2, int dim3, int dim4) {
  if (threadIdx.x == 0) {
    switch (dims) {
      case 0:
        break;
      case 1:
        printf("Shape: (%d)\n", dim0);
        break;
      case 2:
        printf("Shape: (%d, %d)\n", dim0, dim1);
        break;
      case 3:
        printf("Shape: (%d, %d, %d)\n", dim0, dim1, dim2);
        break;
      case 4:
        printf("Shape: (%d, %d, %d, %d)\n", dim0, dim1, dim2, dim3);
        break;
      case 5:
        printf("Shape: (%d, %d, %d, %d, %d)\n", dim0, dim1, dim2, dim3, dim4);
        break;
      default:
        printf("Shape: (%d, %d, %d, %d, %d, ...)\n", dim0, dim1, dim2, dim3, dim4);
        break;
    }
  }
}

constexpr int64_t kDefaultSnippetEdgeItems = 3;
constexpr int64_t kDefaultSnippetThreshold = 200;

// Skip non edge items in last dimension
#define SKIP_NON_EDGE_ITEMS_LAST_DIM(dim_size, index, edge_items)                          \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      printf(", ... ");                                                                    \
    }                                                                                      \
    continue;                                                                              \
  }

// Skip non edge items in other dimensions except the last dimension
#define SKIP_NON_EDGE_ITEMS(dim_size, index, edge_items)                                   \
  if (dim_size > 2 * edge_items && index >= edge_items && index + edge_items < dim_size) { \
    if (index == edge_items) {                                                             \
      printf("...\n");                                                                     \
    }                                                                                      \
    continue;                                                                              \
  }

inline __device__ void PrintValue(const float& value) {
  printf("%.8f", value);
}

inline  __device__ void PrintValue(const half& value) {
  printf("%.8f", __half2float(value));
}

inline  __device__ void PrintValue(const int8_t& value) {
  printf("%d", static_cast<int32_t>(value));
}

inline  __device__ void PrintValue(const int32_t& value) {
  printf("%d", value);
}

inline  __device__ void PrintValue(const int64_t& value) {
  printf("%d", static_cast<int32_t>(value));
}

// Print snippet of 2D tensor with shape (dim0, dim1)
template <typename T>
inline  __device__ void PrintCpuTensorSnippet(const T* tensor, int64_t dim0, int64_t dim1, int64_t edge_items) {
  for (int64_t i = 0; i < dim0; i++) {
    SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
    PrintValue(tensor[i * dim1]);
    for (int64_t j = 1; j < dim1; j++) {
      SKIP_NON_EDGE_ITEMS_LAST_DIM(dim1, j, edge_items);
      printf(", ");
      PrintValue(tensor[i * dim1 + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// Print snippet of 3D tensor with shape (dim0, dim1, dim2)
template <typename T>
inline  __device__ void PrintCpuTensorSnippet(const T* tensor, int64_t dim0, int64_t dim1, int64_t dim2, int64_t edge_items) {
  for (int64_t i = 0; i < dim0; i++) {
    SKIP_NON_EDGE_ITEMS(dim0, i, edge_items);
    for (int64_t j = 0; j < dim1; j++) {
      SKIP_NON_EDGE_ITEMS(dim1, j, edge_items);
      PrintValue(tensor[i * dim1 * dim2 + j * dim2]);
      for (int64_t k = 1; k < dim2; k++) {
        SKIP_NON_EDGE_ITEMS_LAST_DIM(dim2, k, edge_items);
        printf(", ");
        PrintValue(tensor[i * dim1 * dim2 + j * dim2 + k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

// Print 2D tensor
template <typename T>
inline  __device__ void PrintCpuTensorFull(const T* tensor, int64_t dim0, int64_t dim1) {
  for (int64_t i = 0; i < dim0; i++) {
    PrintValue(tensor[i * dim1]);
    for (int64_t j = 1; j < dim1; j++) {
      printf(", ");
      PrintValue(tensor[i * dim1 + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// Print 3D tensor
template <typename T>
inline  __device__ void PrintCpuTensorFull(const T* tensor, int64_t dim0, int64_t dim1, int64_t dim2) {
  for (int64_t i = 0; i < dim0; i++) {
    for (int64_t j = 0; j < dim1; j++) {
      PrintValue(tensor[i * dim1 * dim2 + j * dim2]);
      for (int64_t k = 1; k < dim2; k++) {
        printf(", ");
        PrintValue(tensor[i * dim1 * dim2 + j * dim2 + k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void PrintString(char n0, char n1, char n2, char n3, char n4, char n5, char n6, char n7, char n8, char n9, bool eol) {
  if (threadIdx.x == 0) {
      printf("%c%c%c%c%c%c%c%c%c%c", n0, n1, n2, n3, n4, n5, n6, n7, n8, n9);
      if (eol) printf("\n");
  }
}

char GetChar(const std::string& name, size_t index){
  if (index < name.length()){
    return name[index];
  }
  return ' ';
}

void DumpString(cudaStream_t stream, const std::string& name, bool eol) {
  // If length is larger than 20, only show first 20 characters.
  if (name.length() > 10) {
    PrintString<<<1, 1, 0, stream>>>(
    GetChar(name, 0),
    GetChar(name, 1),
    GetChar(name, 2),
    GetChar(name, 3),
    GetChar(name, 4),
    GetChar(name, 5),
    GetChar(name, 6),
    GetChar(name, 7),
    GetChar(name, 8),
    GetChar(name, 9),
    false
    );

    PrintString<<<1, 1, 0, stream>>>(
    GetChar(name, 10),
    GetChar(name, 11),
    GetChar(name, 12),
    GetChar(name, 13),
    GetChar(name, 14),
    GetChar(name, 15),
    GetChar(name, 16),
    GetChar(name, 17),
    GetChar(name, 18),
    GetChar(name, 19),
    eol
    );
  } else {
    PrintString<<<1, 1, 0, stream>>>(
        GetChar(name, 0),
        GetChar(name, 1),
        GetChar(name, 2),
        GetChar(name, 3),
        GetChar(name, 4),
        GetChar(name, 5),
        GetChar(name, 6),
        GetChar(name, 7),
        GetChar(name, 8),
        GetChar(name, 9),
        eol
        );
  }
}


__global__ void PrintTensorInfo(bool is_input, int index) {
  if (threadIdx.x == 0) {
    if (is_input) {
      printf("Input %d Name:", index);
    } else {
      printf("Output %d Name:", index);
    }
  }
}

template <typename T>
__global__ void Print2DTensor(const T* tensor, int dim0, int dim1, bool snippet, int edge_items) {
  if (threadIdx.x == 0) {
    if (snippet) {
      PrintCpuTensorSnippet<T>(tensor, dim0, dim1, edge_items);
    } else {
      PrintCpuTensorFull<T>(tensor, dim0, dim1);
    }
  }
}

template <typename T>
__global__ void Print3DTensor(const T* tensor, int dim0, int dim1, int dim2, bool snippet, int edge_items) {
  if (threadIdx.x == 0) {
    if (snippet) {
      PrintCpuTensorSnippet<T>(tensor, dim0, dim1, dim2, edge_items);
    } else {
      PrintCpuTensorFull<T>(tensor, dim0, dim1, dim2);
    }
  }
}

template <typename T>
void Dump2DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, bool snippet, int edge_items) {
  Print2DTensor<<<1, 1, 0, stream>>>(tensor, dim0, dim1, snippet, edge_items);
}

template <typename T>
void Dump3DTensor(cudaStream_t stream, const T* tensor, int dim0, int dim1, int dim2, bool snippet, int edge_items) {
  Print3DTensor<<<1, 1, 0, stream>>>(tensor, dim0, dim1, dim2, snippet, edge_items);
}

// template instantiation
template void Dump2DTensor<float>(cudaStream_t, const float*, int, int, bool, int);
template void Dump2DTensor<int8_t>(cudaStream_t, const int8_t*, int, int, bool, int);
template void Dump2DTensor<int32_t>(cudaStream_t, const int32_t*, int, int, bool, int);
template void Dump2DTensor<int64_t>(cudaStream_t, const int64_t*, int, int, bool, int);


template void Dump3DTensor<float>(cudaStream_t, const float*, int, int, int, bool, int);
template void Dump3DTensor<int8_t>(cudaStream_t, const int8_t*, int, int, int, bool, int);
template void Dump3DTensor<int32_t>(cudaStream_t, const int32_t*, int, int, int, bool, int);
template void Dump3DTensor<int64_t>(cudaStream_t, const int64_t*, int, int, int, bool, int);

template<>
void Dump2DTensor(cudaStream_t stream, const MLFloat16* tensor, int dim0, int dim1, bool snippet, int edge_items) {
  Print2DTensor<<<1, 1, 0, stream>>>(reinterpret_cast<const half*>(tensor), dim0, dim1, snippet, edge_items);
}

template<>
void Dump3DTensor(cudaStream_t stream, const MLFloat16* tensor, int dim0, int dim1, int dim2, bool snippet, int edge_items) {
  Print3DTensor<<<1, 1, 0, stream>>>(reinterpret_cast<const half*>(tensor), dim0, dim1, dim2, snippet, edge_items);
}


template <typename T>
void PrintGpuTensor(const Tensor& tensor, cudaStream_t stream, int edge_items, int threshold) {
  const auto& shape = tensor.Shape();
  auto num_items = shape.Size();
  if (num_items == 0) {
    return;
  }

  size_t num_dims = shape.NumDimensions();
  PrintTensorShape<<<1, 1, 0, stream>>>(num_dims,
                                        shape[0],
                                        num_dims > 1 ? shape[1] : 0,
                                        num_dims > 2 ? shape[2] : 0,
                                        num_dims > 3 ? shape[3] : 0,
                                        num_dims > 4 ? shape[4] : 0);

  auto data = tensor.Data<T>();
  bool is_snippet = (threshold > 0 && static_cast<int64_t>(threshold) < num_items);
  if (num_dims >= 3) {
    int dim0 = static_cast<int>(shape.SizeToDimension(num_dims - 2));
    int dim1 = static_cast<int>(shape[num_dims - 2]);
    int dim2 = static_cast<int>(shape[num_dims - 1]);
    Dump3DTensor<T>(stream, data, dim0, dim1, dim2, is_snippet, edge_items);
    return;
  }

  size_t num_rows = 1;
  if (num_dims > 1) {
    num_rows = static_cast<size_t>(shape[0]);
  }
  size_t row_size = num_items / num_rows;
  Dump2DTensor<T>(stream, data, num_rows, row_size, is_snippet, edge_items);
}


void DumpGpuTensor(const Tensor& tensor, cudaStream_t stream) {
  switch (tensor.DataType()->AsPrimitiveDataType()->GetDataType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      PrintGpuTensor<float>(tensor, stream, kDefaultSnippetEdgeItems, kDefaultSnippetThreshold);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      PrintGpuTensor<int8_t>(tensor, stream, kDefaultSnippetEdgeItems, kDefaultSnippetThreshold);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      PrintGpuTensor<int32_t>(tensor, stream, kDefaultSnippetEdgeItems, kDefaultSnippetThreshold);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      PrintGpuTensor<int64_t>(tensor, stream, kDefaultSnippetEdgeItems, kDefaultSnippetThreshold);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      PrintGpuTensor<MLFloat16>(tensor, stream, kDefaultSnippetEdgeItems, kDefaultSnippetThreshold);
      break;
  }
}


void DumpTensor(const Tensor& tensor, cudaStream_t stream) {
  // check tensor is on CPU before dumping it
  auto& tensor_location = tensor.Location();
  if (tensor_location.device.Type() == OrtDevice::CPU ||
      tensor_location.mem_type == OrtMemTypeCPUInput ||
      tensor_location.mem_type == OrtMemTypeCPUOutput) {
    // DumpCpuTensor(tensor);
  } else {
    DumpGpuTensor(tensor, stream);
  }
}

}  // namespace

void CudaKernel::DumpInputs(OpKernelContext* p_op_kernel_context) const{
  const OpKernelContext& context = *p_op_kernel_context;
  const onnxruntime::Node& node = this->Node();
  cudaStream_t stream = Stream(p_op_kernel_context);

  //std::cout << "-----------\n";
  std::string line = "----------";
  DumpString(stream, line, true);

  //std::cout << node.OpType() << " node: " << node.Name() << "\n";
  DumpString(stream, node.OpType(), false);
  std::string title = " Node:";
  DumpString(stream, title, false);
  DumpString(stream, node.Name(), true);

  const auto& input_defs = node.InputDefs();

  for (auto i = 0, end = context.InputCount(); i < end; ++i) {
    if (input_defs[i]->Exists()) {

      const std::string& name = input_defs[i]->Name();
      PrintTensorInfo<<<1, 1, 0, stream>>>(true, i);
      DumpString(stream, name, true);

      const auto* type = context.InputType(i);
      //const auto* type = input_defs[i]->TypeAsProto();
      //MLDataType type = input_defs[i]->TypeAsProto();
      if (type) {
        if (type->IsTensorType()) {
          const auto* tensor = context.Input<Tensor>(i);
          if (tensor != nullptr) {
            DumpTensor(*tensor, stream);
          }
        }
      }
    }
  }
}



void CudaKernel::DumpOutputs(OpKernelContext* p_op_kernel_context) const{
  OpKernelContext& context = *p_op_kernel_context;
  const onnxruntime::Node& node = this->Node();
  cudaStream_t stream = Stream(p_op_kernel_context);

  //std::cout << "-----------\n";
  std::string line = "----------";
  DumpString(stream, line, true);

  const auto& output_defs = node.OutputDefs();
  for (auto i = 0, end = context.OutputCount(); i < end; ++i) {
    if (output_defs[i]->Exists()) {

      const std::string& name = output_defs[i]->Name();
      PrintTensorInfo<<<1, 1, 0, stream>>>(false, i);
      DumpString(stream, name, true);

      const auto* type = context.OutputType(i);
      if (type) {
        if (type->IsTensorType()) {
          const auto* tensor = context.Output<Tensor>(i);
          if (tensor != nullptr) {
            DumpTensor(*tensor, stream);
          }
        }
      }
    }

    std::cout << std::endl;
  }
}
#endif

}
}
