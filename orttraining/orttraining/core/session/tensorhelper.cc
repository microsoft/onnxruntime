// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/session/tensorhelper.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::experimental;
using namespace onnxruntime::common;
namespace onnxruntime {

  size_t findIndex(std::vector<int64_t> dims, std::vector<int64_t> indices){
    size_t linear_index = 0;
    for(unsigned int i = 0 ; i < dims.size(); i++){
      linear_index = indices[i] + dims[i]*(linear_index);
    }
    return linear_index;
  }

  OrtValue SliceTensor(const OrtValue& orig_value, const size_t slice_id,
                      const size_t slice_axis, const size_t num_slices, TrainingSession& session_state) {      
      
      // Get tensor from OrtValue
      const Tensor& orig_tensor = orig_value.Get<Tensor>();
      // Get the tensor shape
      const TensorShape& orig_tensor_shape = orig_tensor.Shape();
      const MLDataType orig_tensor_type = orig_tensor.DataType();
      const OrtMemoryInfo orig_tensor_location = orig_tensor.Location(); 
              
      std::vector<int64_t> orig_dims = orig_tensor_shape.GetDims(); // dims has the tensor axis
      std::vector<int64_t> small_dims; // make a vector to store dims of the sliced tensor
      if(orig_dims[slice_axis] % num_slices != 0){
        std::cout<<"<ERROR>: num_slices does not evenly divide the slice_axis" << ", orig_dims[slice_axis] "<< orig_dims[slice_axis] << ", num_slices "<< num_slices<< std::endl;
        ORT_ENFORCE(false, "This shouldn't happen.");
      }
      //size_t slice_size = dims[slice_axis]/num_slices; 
      size_t contiguous_slice_size = 1;
      size_t total_num_elements = 1;
      for (size_t i_shape = 0; i_shape < orig_dims.size(); ++i_shape) {
        if(i_shape == slice_axis){
          small_dims.push_back(orig_dims[i_shape]/num_slices);
        }else{
          small_dims.push_back(orig_dims[i_shape]);
        }
        if(i_shape > slice_axis){
          contiguous_slice_size *= orig_dims[i_shape];
        }
        total_num_elements *= orig_dims[i_shape];
        
      }
        
      TensorShape small_shape(small_dims); 

      std::cout << "[inference_session.cc] Old shape " << orig_tensor_shape << std::endl;
      std::cout << "[inference_session.cc] Small shape " << small_shape << std::endl;

      OrtMemoryInfo cpu_location(onnxruntime::CPU, OrtDeviceAllocator); // ??
      AllocatorPtr cpu_allocator = session_state.GetAllocator(cpu_location);
      auto small_cpu_tensor = onnxruntime::make_unique<Tensor>(orig_tensor_type, small_shape, cpu_allocator); // make tensor variable along the CPU
      auto tensor_type = DataTypeImpl::GetType<Tensor>();
      OrtValue small_cpu_value{small_cpu_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()}; //
      auto cpu_ptr = orig_value.Get<Tensor>().DataRaw(); //get cpu pointer from original tensor
      auto small_cpu_ptr = small_cpu_value.GetMutable<Tensor>()->MutableDataRaw(); // ???..


      int device; 
      cudaGetDevice(&device);
      //size_t bias =  round * old_type->Size() * small_shape.Size(); // how was bias calculated ?
      //size_t copied_size = old_type->Size() * small_shape.Size(); // what is copied size  ??
      // slice_id, axis, numslices
      size_t elements_read_so_far = 0;
      size_t bias  = 0;
      size_t num_strides =0;
      size_t slice_size = orig_dims[slice_axis]/num_slices; //assuming that dims[slice_axis] is perfectly divisible else throw error
      size_t copied_size = orig_tensor_type->Size() * contiguous_slice_size *slice_size;
      while(elements_read_so_far <= total_num_elements){
      
        // slice_size * slice_id will give us number of elements to copy inside the axis dimension          
        //copied_size = orig_tensor_type->Size() * contiguous_slice_size *slice_size;
                    
        bias = (elements_read_so_far + slice_id * slice_size * contiguous_slice_size) * orig_tensor_type->Size();  
        if (std::string(orig_tensor_location.name) == std::string("Cuda")) {
          if (device != orig_tensor_location.id) {
            cudaSetDevice(orig_tensor_location.id);
          }

          cudaMemcpy(static_cast<char*>((void*)small_cpu_ptr) + num_strides*copied_size, static_cast<const char*>(cpu_ptr) + bias, copied_size, cudaMemcpyDeviceToHost);                  

          if (device != orig_tensor_location.id) {
            cudaSetDevice(device);
          }
        } else {
          memcpy(static_cast<char*>((void*)small_cpu_ptr) + num_strides*copied_size, static_cast<const char*>(cpu_ptr) + bias, copied_size);                  
        }
        //bias = (elements_read_so_far + slice_id*slice_size*contiguous_slice_size);
        //if (orig_tensor_type->Size() == 4) {
        //  std::cout << "-----------------------------" << std::endl;
        //  std::cout << "Do float copy" << std::endl;
        //  std::cout << "-----------------------------" << std::endl;
        //  // Assume type is float.
        //  const float* ptr_from = reinterpret_cast<const float*>(cpu_ptr);
        //  float* ptr_to = reinterpret_cast<float*>(small_cpu_ptr);
        //  for (size_t ii = 0; ii < contiguous_slice_size * slice_size; ++ii) {
        //    ptr_to[num_strides * contiguous_slice_size * slice_size + ii] = ptr_from[bias + ii];
        //  }
        //} else {
        //  // Assume type is int64.
        //  std::cout << "-----------------------------" << std::endl;
        //  std::cout << "Do int64 copy" << std::endl;
        //  std::cout << "-----------------------------" << std::endl;
        //  const int64_t* ptr_from = reinterpret_cast<const int64_t*>(cpu_ptr);
        //  int64_t* ptr_to = reinterpret_cast<int64_t*>(small_cpu_ptr);
        //  for (size_t ii = 0; ii < contiguous_slice_size * slice_size; ++ii) {
        //    ptr_to[num_strides * contiguous_slice_size * slice_size + ii] = ptr_from[bias + ii];
        //  }
        //}
        elements_read_so_far += orig_dims[slice_axis] * contiguous_slice_size;
      
        num_strides += 1;
      }

      copied_size = num_strides*copied_size;
      std::cout << "[inference_session.cc] current GPU device " << device << std::endl;;
      if (std::string(orig_tensor_location.name) == std::string("Cuda")) {
        // Get CPU tensor to be copied.
        const Tensor& copied_tensor = small_cpu_value.Get<Tensor>();

        // Create GPU tensor to capture CPU data.
        AllocatorPtr allocator = session_state.GetAllocator(orig_tensor_location);
        auto small_tensor = onnxruntime::make_unique<Tensor>(orig_tensor_type, small_shape, allocator);
        OrtValue small_value{small_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()}; 
        Tensor* capturing_tensor = small_value.GetMutable<Tensor>();

        if (device != orig_tensor_location.id) {
          cudaSetDevice(orig_tensor_location.id);
        }

        cudaMemcpy(capturing_tensor->MutableDataRaw(), copied_tensor.DataRaw(), copied_size, cudaMemcpyHostToDevice);

        if (device != orig_tensor_location.id) {
          cudaSetDevice(device);
        }
        return small_value;
        //new_feeds.push_back(small_value);
      } else if (std::string(orig_tensor_location.name) == std::string("Cpu")) {
        const Tensor& copied_tensor = small_cpu_value.Get<Tensor>();

        AllocatorPtr allocator = session_state.GetAllocator(orig_tensor_location);
        auto small_tensor = onnxruntime::make_unique<Tensor>(orig_tensor_type, small_shape, allocator);
        OrtValue small_value{small_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()}; 
        Tensor* capturing_tensor = small_value.GetMutable<Tensor>();

        memcpy(capturing_tensor->MutableDataRaw(), copied_tensor.DataRaw(), copied_size);
        //new_feeds.push_back(small_value);
        return small_value;
      } else {
        ORT_ENFORCE(false, "This shouldn't happen.");
      }
      
  } // slice func ends

  // concat tensors
  OrtValue ConcatTensor(const std::vector<OrtValue>& orig_values, const size_t n, const size_t axis, TrainingSession& session_state){
    if(orig_values.size() == 0){
      ORT_ENFORCE(false, "the input tensors array is empty"); 
    }
    if(n != orig_values.size()){
      ORT_ENFORCE(false, "number of tensors in array not equal to target number of sub-batches"); 
    }
    const Tensor& inp_tensor = orig_values.at(0).Get<Tensor>();
    const TensorShape& inp_tensor_shape = inp_tensor.Shape();
    const MLDataType inp_tensor_type = inp_tensor.DataType();
    const OrtMemoryInfo inp_tensor_location = inp_tensor.Location();
    // allocate a pointer for concated tensor
    std::vector<int64_t> concatenated_dims;
    size_t contiguous_slice_size = 1;
    size_t total_num_elements = 1;
    std::vector<int64_t> inp_dims = inp_tensor_shape.GetDims();
    for (size_t i_shape = 0; i_shape < inp_dims.size(); ++i_shape) {
      if(i_shape == axis){
        concatenated_dims.push_back(inp_dims[i_shape]*n);
      }else{
        concatenated_dims.push_back(inp_dims[i_shape]);
      }
      if(i_shape >= axis){
          contiguous_slice_size *= inp_dims[i_shape]; // includes the size of current chunk (batch/n)
      }
      total_num_elements *= inp_dims[i_shape];    
    }
    total_num_elements *= n; // size of concatenated tensor is total elements of one batch * number of batches
    TensorShape concatenated_shape(concatenated_dims); 
    // allocate a pointer for concatenated tensor
    int device; 
    cudaGetDevice(&device);
    OrtMemoryInfo cpu_location(onnxruntime::CPU, OrtDeviceAllocator); // ??
    AllocatorPtr cpu_allocator = session_state.GetAllocator(cpu_location);
    auto concatenated_cpu_tensor = onnxruntime::make_unique<Tensor>(inp_tensor_type, concatenated_shape, cpu_allocator); 

    auto tensor_type = DataTypeImpl::GetType<Tensor>();
    OrtValue concatenated_cpu_value{concatenated_cpu_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()}; //
    auto concatenated_cpu_ptr = concatenated_cpu_value.GetMutable<Tensor>()->MutableDataRaw(); // ???..

    size_t elements_read_so_far = 0; // number of elements copied to concatenated tensor
    size_t num_strides = 0;
    while(elements_read_so_far <= total_num_elements){
      for(size_t i = 0; i < orig_values.size(); ++i){
        auto& orig_value = orig_values.at(i); 
        // Get tensor from OrtValue
        const Tensor& orig_tensor = orig_value.Get<Tensor>();
        // Get the tensor shape
        //const TensorShape& orig_tensor_shape = orig_tensor.Shape();
        const MLDataType orig_tensor_type = orig_tensor.DataType();
        const OrtMemoryInfo orig_tensor_location = orig_tensor.Location();

        auto orig_ptr = orig_value.Get<Tensor>().DataRaw(); //get cpu or gpu pointer from original tensor

        size_t copied_size = contiguous_slice_size*orig_tensor_type->Size();//number of slices
        size_t bias = (contiguous_slice_size * num_strides) * orig_tensor_type->Size();  
        if (std::string(orig_tensor_location.name) == std::string("Cuda")) {
          if (device != orig_tensor_location.id) {
            cudaSetDevice(orig_tensor_location.id);
          }

          cudaMemcpy(static_cast<char*>((void*)concatenated_cpu_ptr) + elements_read_so_far*orig_tensor_type->Size(), static_cast<const char*>(orig_ptr) + bias, copied_size, cudaMemcpyDeviceToHost);                  

          if (device != orig_tensor_location.id) {
            cudaSetDevice(device);
          }
        } else {
          memcpy(static_cast<char*>((void*)concatenated_cpu_ptr) + elements_read_so_far*orig_tensor_type->Size(), static_cast<const char*>(orig_ptr) + bias, copied_size);                  
        }
        elements_read_so_far += contiguous_slice_size;
      }
      num_strides += 1;

    } // while ends

    size_t copied_size = elements_read_so_far*inp_tensor_type->Size();
    std::cout << "final copied size =  " << copied_size << ", elements_read_so_far = " << elements_read_so_far << ", orig_tensor_type->Size() = " << inp_tensor_type->Size() << std::endl;;
    std::cout << "[inference_session.cc] current GPU device " << device << std::endl;;
    if (std::string(inp_tensor_location.name) == std::string("Cuda")) {
      // Get CPU tensor to be copied.
      const Tensor& copied_tensor = concatenated_cpu_value.Get<Tensor>();

      // Create GPU tensor to capture CPU data.
      AllocatorPtr allocator = session_state.GetAllocator(inp_tensor_location);
      auto concatenated_tensor = onnxruntime::make_unique<Tensor>(inp_tensor_type, concatenated_shape, allocator);
      OrtValue concatenated_value{concatenated_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()}; 
      Tensor* capturing_tensor = concatenated_value.GetMutable<Tensor>();

      if (device != inp_tensor_location.id) {
        cudaSetDevice(inp_tensor_location.id);
      }

      cudaMemcpy(capturing_tensor->MutableDataRaw(), copied_tensor.DataRaw(), copied_size, cudaMemcpyHostToDevice);

      if (device != inp_tensor_location.id) {
        cudaSetDevice(device);
      }
      return concatenated_value;
      //new_feeds.push_back(small_value);
    } else if (std::string(inp_tensor_location.name) == std::string("Cpu")) {
      const Tensor& copied_tensor = concatenated_cpu_value.Get<Tensor>();

      AllocatorPtr allocator = session_state.GetAllocator(inp_tensor_location);
      auto concatenated_tensor = onnxruntime::make_unique<Tensor>(inp_tensor_type, concatenated_shape, allocator);
      OrtValue concatenated_value{concatenated_tensor.release(), tensor_type, tensor_type->GetDeleteFunc()}; 
      Tensor* capturing_tensor = concatenated_value.GetMutable<Tensor>();

      memcpy(capturing_tensor->MutableDataRaw(), copied_tensor.DataRaw(), copied_size);
      //new_feeds.push_back(small_value);
      return concatenated_value;
    } else {
      ORT_ENFORCE(false, "This shouldn't happen.");
    }

  } // ConcatTensor func ends
  
}