//
//  OnnxInterop.cpp
//  SwiftMnist
//
//  Created by Miguel de Icaza on 6/1/20.
//  Copyright © 2020 Miguel de Icaza. All rights reserved.
//
#include <array>
#include <onnxruntime_cxx_api.h>
extern "C" {
#include "SwiftMnist-Bridging-Header.h"
}
struct MNIST {
  MNIST() {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
  }
    
  std::ptrdiff_t Run() {
    const char* input_names[] = {"Input3"};
    const char* output_names[] = {"Plus214_Output_0"};
    
    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }
  
  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;
  
  std::array<float, width_ * height_> input_image_{};
  std::array<float, 10> results_{};
  int64_t result_{0};
  
 private:
  Ort::Env env;
  Ort::Session session_{env, "model.onnx", Ort::SessionOptions{nullptr}};
    
  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};
};
    
mnist *mnist_new ()
{
    return (mnist *) new MNIST();
}

float *mnist_get_input_image (mnist *_mnist, size_t *out)
{
    MNIST *mnist = (MNIST *) _mnist;
    *out = mnist->input_image_.size();
    return mnist->input_image_.data ();
}

float *mnist_get_results (mnist *_mnist, size_t *out)
{
    MNIST *mnist = (MNIST *) _mnist;
    *out = mnist->results_.size();
    return mnist->results_.data ();
}

long mnist_run (mnist *_mnist)
{
    MNIST *mnist = (MNIST *) _mnist;

    mnist->Run();
    return mnist->result_;
}
