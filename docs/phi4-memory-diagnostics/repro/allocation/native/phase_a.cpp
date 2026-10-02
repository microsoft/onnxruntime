// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cmath>

#include "capture_support.h"

void SaveLogits(OgaTensor& tensor, const fs::path& directory) {
  Require(tensor.Type() == OgaElementType_float32 &&
              tensor.Shape() == std::vector<int64_t>({1, 1, 200064}),
          "Unexpected logits dtype or shape");
  size_t finite = 0, nan = 0, positive_infinity = 0, negative_infinity = 0;
  const auto* values = static_cast<const float*>(tensor.Data());
  for (size_t i = 0; i < 200064; ++i) {
    if (std::isfinite(values[i]))
      ++finite;
    else if (std::isnan(values[i]))
      ++nan;
    else if (values[i] > 0)
      ++positive_infinity;
    else
      ++negative_infinity;
  }
  Require(finite >= 2 && nan == 0, "Invalid full prefill logits");
  WriteBinary(directory / "logits.bin", values, 800256);
  Write(directory / "logits.json",
        "{\"dtype\":\"float32\",\"shape\":[1,1,200064],\"element_count\":200064,\"byte_count\":800256,"
        "\"byte_order\":\"little\",\"format\":\"headerless contiguous row-major binary\","
        "\"finite_count\":" +
            std::to_string(finite) + ",\"nan_count\":" + std::to_string(nan) +
            ",\"positive_infinity_count\":" + std::to_string(positive_infinity) +
            ",\"negative_infinity_count\":" + std::to_string(negative_infinity) + "}\n");
}

int main(int argc, char** argv) {
  try {
    const auto ids = LoadInput(argc, argv);
    const fs::path directory = argv[6];
    OgaHandle shutdown;
    std::cout << "[ ORT ] " << OrtGetApiBase()->GetVersionString() << std::endl;
    auto config = OgaConfig::Create(argv[1]);
    config->ClearProviders();
    config->AppendProvider("CUDA");
    config->SetProviderOption("CUDA", "device_id", "0");
    const auto effective = Read(argv[7]);
    config->Overlay(effective.c_str());
    Write(directory / "applied-config.json", effective);
    auto model = OgaModel::Create(*config);
    const auto device = model->GetDeviceType();
    Require(std::string(static_cast<const char*>(device)) == "CUDA", "Model is not CUDA");
    VerifyLibraries(argv[2], argv[3]);
    auto params = OgaGeneratorParams::Create(*model);
    ConfigureSearch(*params, ids.size(), directory);
    auto generator = OgaGenerator::Create(*model, *params);
    Write(directory / "loaded-maps-before.txt", Read("/proc/self/maps"));
    AwaitSampler("ready", "GO");
    generator->AppendTokens(ids.data(), ids.size());
    Require(generator->GetSequenceCount(0) == ids.size(), "Unexpected history length after append");
    auto logits = generator->GetLogits();
    SaveLogits(*logits, directory);
    logits.reset();
    Require(generator->GetSequenceCount(0) == ids.size(), "Logits inspection changed history length");
    Require(std::equal(ids.begin(), ids.end(), generator->GetSequenceData(0)), "History IDs changed");
    for (size_t i = 0; i < 64; ++i) {
      Require(!generator->IsDone(), "Generation ended before the required 64 tokens");
      generator->GenerateNextToken();
      Require(generator->GetSequenceCount(0) == ids.size() + i + 1, "Unexpected generation sequence length");
    }
    SaveTokens(*generator, ids, directory);
    VerifyLibraries(argv[2], argv[3]);
    Write(directory / "loaded-maps-after.txt", Read("/proc/self/maps"));
    SaveState(directory, std::stoi(argv[5]), ids.size(), false);
    std::cout << "[ SUCCESS ] Phase A correctness capture" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ FAILURE ] " << error.what() << std::endl;
    return 1;
  }
}
