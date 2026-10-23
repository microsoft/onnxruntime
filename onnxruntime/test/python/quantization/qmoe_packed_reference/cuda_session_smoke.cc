// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const OrtApi* api = OrtGetApiBase()->GetApi(31);
  if (!api) return 2;
  Ort::InitApi(api);
  const bool expect_failure = std::string(argv[2]) == "invalid";
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "int3-session-smoke");
    Ort::SessionOptions options;
    OrtCUDAProviderOptions cuda_options{};
    options.AppendExecutionProvider_CUDA(cuda_options);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.AddConfigEntry("ep.cuda.qmoe_row_tile_size", "2");
    Ort::Session session(env, argv[1], options);
    for (int repeat = 0; repeat < 3; ++repeat) {
      const char* output_name = "output";
      auto outputs = session.Run(Ort::RunOptions{nullptr}, nullptr, nullptr, 0, &output_name, 1);
      if (expect_failure) {
        std::fprintf(stderr, "FAIL: malformed INT3 content was accepted\n");
        return 1;
      }
      const auto info = outputs.front().GetTensorTypeAndShapeInfo();
      std::vector<float> expected_values;
      const bool has_reference = argv[2][0] == '@';
      if (has_reference) {
        expected_values.resize(info.GetElementCount());
        std::ifstream reference(argv[2] + 1, std::ios::binary);
        if (!reference.read(reinterpret_cast<char*>(expected_values.data()), expected_values.size() * sizeof(float))) return 2;
      }
      const float constant_expected = has_reference ? 0 : std::stof(argv[2]);
      const bool bfloat = info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
      const auto* bytes = outputs.front().GetTensorData<uint16_t>();
      for (size_t index = 0; index < info.GetElementCount(); ++index) {
        const float expected = has_reference ? expected_values[index] : constant_expected;
        const float actual = bfloat ? float(Ort::BFloat16_t::FromBits(bytes[index]))
                                    : float(Ort::Float16_t::FromBits(bytes[index]));
        if (!std::isfinite(actual) || std::abs(actual - expected) > 0.005f) {
          std::fprintf(stderr, "FAIL index=%zu expected=%g actual=%g\n", index, expected, actual);
          return 1;
        }
      }
    }
    std::printf("PASS: CUDA-only Session, three repeated calls, %s\n", argv[1]);
    return 0;
  } catch (const Ort::Exception& error) {
    if (expect_failure && std::string(error.what()).find("finite nonnegative scales and INT3 zero points") != std::string::npos) {
      std::printf("PASS: malformed GPU content rejected, %s\n", argv[1]);
      return 0;
    }
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
