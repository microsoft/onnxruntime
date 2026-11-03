#include <onnxruntime_cxx_api.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "hc_original_reference_io.inc"

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("Expected an explicit temporary reference path");
    const std::filesystem::path path{argv[1]};
    if (std::filesystem::exists(path)) throw std::runtime_error("Existing test reference");
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "reference-io-test"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Float16_t halves[] = {Ort::Float16_t(0.0f), Ort::Float16_t(-0.0f),
                               Ort::Float16_t(1.0f), Ort::Float16_t(-2.0f)};
    float floats[] = {0.0f, -0.0f, 0.125f, -23.25f};
    const int64_t shape[] = {2, 2};
    auto half = Ort::Value::CreateTensor<Ort::Float16_t>(memory, halves, 4, shape, 2);
    auto single = Ort::Value::CreateTensor<float>(memory, floats, 4, shape, 2);
    {
      std::fstream stream(path, std::ios::binary | std::ios::out);
      TransferOriginalOutput(stream, true, half, "half-write");
      TransferOriginalOutput(stream, true, single, "float-write");
      stream.close();
      if (stream.fail()) throw std::runtime_error("Cannot close test writer");
    }
    {
      std::fstream stream(path, std::ios::binary | std::ios::in);
      TransferOriginalOutput(stream, false, half, "half-read");
      TransferOriginalOutput(stream, false, single, "float-read");
      if (stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Unexpected trailing reference data");
    }
    halves[1] = Ort::Float16_t(0.0f);
    bool zero_rejected = false;
    try {
      std::fstream stream(path, std::ios::binary | std::ios::in);
      TransferOriginalOutput(stream, false, half, "signed-zero");
    } catch (const NumericMismatch&) {
      zero_rejected = true;
    }
    if (!zero_rejected) throw std::runtime_error("Signed-zero mismatch accepted");
    halves[1] = Ort::Float16_t(-0.0f);
    const int64_t wrong_shape[] = {1, 4};
    auto reshaped = Ort::Value::CreateTensor<Ort::Float16_t>(memory, halves, 4, wrong_shape, 2);
    bool shape_rejected = false;
    try {
      std::fstream stream(path, std::ios::binary | std::ios::in);
      TransferOriginalOutput(stream, false, reshaped, "shape");
    } catch (const NumericMismatch&) {
      shape_rejected = true;
    }
    if (!shape_rejected) throw std::runtime_error("Metadata mismatch accepted");
    std::filesystem::resize_file(path, 16);
    bool truncation_rejected = false;
    try {
      std::fstream stream(path, std::ios::binary | std::ios::in);
      TransferOriginalOutput(stream, false, half, "truncated");
    } catch (const std::runtime_error& error) {
      truncation_rejected = std::string(error.what()).find("Truncated original metadata") != std::string::npos;
    }
    if (!truncation_rejected) throw std::runtime_error("Truncated reference accepted");
    std::filesystem::remove(path);
    std::cout << "PASS reference_io_float16_float32_signed_zero_shape_truncation\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << "\n";
    return 1;
  }
}
