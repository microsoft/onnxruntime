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
      const Tensor& orig_tensor = orig_value.Get<Tensor>();;
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
                    
        bias = (elements_read_so_far + slice_id*slice_size*contiguous_slice_size) * orig_tensor_type->Size();  
        memcpy(static_cast<char*>((void*)small_cpu_ptr) + num_strides*copied_size, static_cast<const char*>(cpu_ptr) + bias, copied_size);                  
        elements_read_so_far += orig_dims[slice_axis] * contiguous_slice_size;
      
        num_strides += 1;
      }

      copied_size = num_strides*copied_size;
      int device; 
      cudaGetDevice(&device); 	
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

  
}