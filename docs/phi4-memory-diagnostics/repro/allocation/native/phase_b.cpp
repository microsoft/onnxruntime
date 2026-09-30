// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdlib>
#include <sstream>
#include <time.h>

#include "cuda_runtime_api.h"
#include "capture_support.h"
#include "onnxruntime_experimental_cxx_api.h"

int64_t MonotonicNs() {
  timespec value{};
  Require(clock_gettime(CLOCK_MONOTONIC, &value) == 0, "Cannot read monotonic clock");
  return int64_t{value.tv_sec} * 1000000000 + value.tv_nsec;
}

void Synchronize() {
  const auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(status));
}

template <typename Diagnostic>
void Snapshot(Diagnostic diagnostic, const fs::path& directory, const char* checkpoint) {
  int64_t reclaimed = -1;
  size_t arenas = 0;
  std::ostringstream capture;
  auto* original = std::cout.rdbuf(capture.rdbuf());
  OrtStatus* status = nullptr;
  const auto start = MonotonicNs();
  try {
    status = diagnostic(checkpoint, false, &reclaimed, &arenas);
  } catch (...) {
    std::cout.rdbuf(original);
    std::cout << capture.str() << std::flush;
    throw;
  }
  const auto end = MonotonicNs();
  std::cout.rdbuf(original);
  std::cout << capture.str() << std::flush;
  Ort::ThrowOnError(status);
  Require(reclaimed == 0 && arenas > 0, "Snapshot reclaimed memory or reported no arenas");
  Write(directory / "snapshot-calls.jsonl",
        "{\"checkpoint\":\"" + std::string(checkpoint) +
            "\",\"shrink\":false,\"reclaimed_bytes\":0,"
            "\"arena_count\":" +
            std::to_string(arenas) + ",\"start_ns\":" + std::to_string(start) +
            ",\"end_ns\":" + std::to_string(end) + "}\n",
        true);
}

int main(int argc, char** argv) {
  try {
    const auto ids = LoadInput(argc, argv);
    const fs::path directory = argv[6];
    const char* registration = std::getenv("ORT_ARENA_DIAGNOSTICS");
    Require(registration && std::string(registration) == "1", "Arena registration must be enabled");
    OgaHandle shutdown;
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    Require(api != nullptr, "Required ORT API unavailable");
    auto diagnostic = Ort::Experimental::Get_OrtApi_DebugLogAndShrinkGpuArenas_SinceV29_FnOrThrow(api);
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
    Synchronize();
    Snapshot(diagnostic, directory, "post_initialize");
    auto params = OgaGeneratorParams::Create(*model);
    ConfigureSearch(*params, ids.size(), directory);
    auto generator = OgaGenerator::Create(*model, *params);
    Synchronize();
    Write(directory / "loaded-maps-before.txt", Read("/proc/self/maps"));
    AwaitSampler("ready", "GO");
    const int64_t start = MonotonicNs();
    generator->AppendTokens(ids.data(), ids.size());
    Require(!generator->IsDone(), "Generator ended before the first token");
    generator->GenerateNextToken();
    const int64_t first_sync_start = MonotonicNs();
    Synchronize();
    const int64_t first_token_ns = MonotonicNs();
    size_t steps = 1;
    while (!generator->IsDone() && steps < 64) {
      generator->GenerateNextToken();
      ++steps;
    }
    const int64_t final_sync_start = MonotonicNs();
    Synchronize();
    const int64_t end = MonotonicNs();
    Write(directory / "timing.json",
          "{\"clock\":\"CLOCK_MONOTONIC\",\"start_ns\":" + std::to_string(start) +
              ",\"first_token_ns\":" + std::to_string(first_token_ns) + ",\"end_ns\":" + std::to_string(end) +
              ",\"first_sync_start_ns\":" + std::to_string(first_sync_start) +
              ",\"final_sync_start_ns\":" + std::to_string(final_sync_start) +
              ",\"generation_steps\":" + std::to_string(steps) + "}\n");
    AwaitSampler("ended", "ACK");
    Require(steps == 64, "Generation ended before the required 64 tokens");
    Require(generator->IsDone(), "Generator did not stop at max_length");
    Snapshot(diagnostic, directory, "post_generation");
    SaveTokens(*generator, ids, directory);
    generator.reset();
    params.reset();
    Synchronize();
    Snapshot(diagnostic, directory, "post_generator_cleanup");
    Require(model != nullptr, "Model must remain alive for cleanup snapshot");
    VerifyLibraries(argv[2], argv[3]);
    Write(directory / "loaded-maps-after.txt", Read("/proc/self/maps"));
    SaveState(directory, std::stoi(argv[5]), ids.size(), true);
    std::cout << "[ SUCCESS ] Phase B measurement; generated_tokens=64 shrink_calls=0" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ FAILURE ] " << error.what() << std::endl;
    return 1;
  }
}
