// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "ort_genai.h"

namespace fs = std::filesystem;

inline void Require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

inline std::string Read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  Require(input.is_open(), "Cannot read " + path.string());
  std::string text{std::istreambuf_iterator<char>(input), {}};
  Require(!input.bad(), "Cannot finish reading " + path.string());
  return text;
}

inline void Write(const fs::path& path, const std::string& text, bool append = false) {
  if (!append) Require(!fs::exists(path), "Refusing to overwrite " + path.string());
  std::ofstream output(path, append ? std::ios::app : std::ios::out);
  Require(output.is_open(), "Cannot write " + path.string());
  output << text;
  output.close();
  Require(!output.fail(), "Cannot finish writing " + path.string());
}

inline void WriteBinary(const fs::path& path, const void* data, size_t bytes) {
  Require(!fs::exists(path), "Refusing to overwrite " + path.string());
  std::ofstream output(path, std::ios::binary);
  Require(output.is_open(), "Cannot open " + path.string());
  output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  output.close();
  Require(!output.fail(), "Failed to write " + path.string());
}

inline void VerifyLibraries(const fs::path& ort_home, const fs::path& genai_dir) {
  const std::map<std::string, fs::path> expected{
      {"libonnxruntime.so", ort_home / "lib/libonnxruntime.so"},
      {"libonnxruntime-genai.so", genai_dir / "libonnxruntime-genai.so"},
      {"libonnxruntime-genai-cuda.so", genai_dir / "libonnxruntime-genai-cuda.so"},
      {"libonnxruntime_providers_cuda.so", ort_home / "lib/libonnxruntime_providers_cuda.so"},
      {"libonnxruntime_providers_shared.so", ort_home / "lib/libonnxruntime_providers_shared.so"}};
  std::ifstream maps("/proc/self/maps");
  Require(maps.is_open(), "Cannot read loaded libraries");
  std::map<std::string, std::set<fs::path>> loaded;
  std::string line;
  while (std::getline(maps, line)) {
    const auto start = line.find('/');
    if (start == std::string::npos) continue;
    const fs::path path = line.substr(start);
    const std::string name = path.filename().string();
    for (const auto& [prefix, target] : expected)
      if (name == prefix || name.starts_with(prefix + "."))
        loaded[prefix].insert(fs::canonical(path));
  }
  for (const auto& [name, target] : expected) {
    Require(loaded[name].size() == 1, "Missing or multiple library images: " + name);
    Require(*loaded[name].begin() == fs::canonical(target), "Unexpected library: " + name);
    std::cout << "[ LIBRARY VERIFIED ] " << name << "=" << *loaded[name].begin() << std::endl;
  }
}

inline std::vector<int32_t> LoadInput(int argc, char** argv) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  Require(argc == 8, "Usage: APP MODEL ORT_HOME GENAI_LIB INPUT_I32LE CONTEXT OUTPUT_DIR EFFECTIVE_CONFIG");
  const int context = std::stoi(argv[5]);
  Require(context == 2048 || context == 32768, "Unsupported nominal context");
  const size_t count = context == 2048 ? 1801 : 28747;
  Require(fs::is_directory(argv[6]), "Output directory does not exist");
  Require(fs::file_size(argv[4]) == count * sizeof(int32_t), "Unexpected serialized input size");
  std::vector<int32_t> ids(count);
  std::ifstream input(argv[4], std::ios::binary);
  input.read(reinterpret_cast<char*>(ids.data()), ids.size() * sizeof(int32_t));
  Require(static_cast<bool>(input), "Failed reading saved input IDs");
  Require(std::all_of(ids.begin(), ids.end(), [](int32_t id) { return id >= 0 && id < 200064; }),
          "Input ID outside vocabulary");
  WriteBinary(fs::path(argv[6]) / "loaded-input-ids.i32le", ids.data(), ids.size() * sizeof(int32_t));
  return ids;
}

inline void ConfigureSearch(OgaGeneratorParams& params, size_t count, const fs::path& directory) {
  params.SetSearchOption("max_length", count + 64);
  params.SetSearchOption("batch_size", 1);
  params.SetSearchOption("chunk_size", 0);
  std::ofstream search(directory / "search-readback.json");
  Require(search.is_open(), "Cannot save search readback");
  search << "{";
  bool first = true;
  for (const char* key : {"max_length", "batch_size", "chunk_size", "num_beams",
                          "num_return_sequences", "top_k", "top_p", "temperature",
                          "min_length", "repetition_penalty", "no_repeat_ngram_size",
                          "diversity_penalty", "length_penalty"}) {
    if (!first) search << ",";
    first = false;
    search << "\"" << key << "\":" << params.GetSearchNumber(key);
  }
  search << ",\"do_sample\":" << (params.GetSearchBool("do_sample") ? "true" : "false")
         << ",\"early_stopping\":" << (params.GetSearchBool("early_stopping") ? "true" : "false")
         << ",\"past_present_share_buffer\":"
         << (params.GetSearchBool("past_present_share_buffer") ? "true" : "false") << "}\n";
  search.close();
  Require(!search.fail(), "Search readback write failed");
  Require(params.GetSearchNumber("max_length") == count + 64 &&
              params.GetSearchNumber("chunk_size") == 0 &&
              params.GetSearchNumber("batch_size") == 1 &&
              params.GetSearchNumber("num_beams") == 1 &&
              !params.GetSearchBool("do_sample"),
          "Search settings mismatch");
}

inline void AwaitSampler(const char* event, const char* response) {
  std::cout << "[ CONTROL ] " << event << std::endl;
  std::string permission;
  Require(static_cast<bool>(std::getline(std::cin, permission)) && permission == response,
          "Missing sampler authorization");
}

inline void SaveTokens(OgaGenerator& generator, const std::vector<int32_t>& ids, const fs::path& directory) {
  Require(generator.IsDone(), "Generator did not stop at max_length");
  Require(generator.GetSequenceCount(0) == ids.size() + 64, "Unexpected sequence length");
  const auto* sequence = generator.GetSequenceData(0);
  Require(std::equal(ids.begin(), ids.end(), sequence), "Generated history changed input");
  const std::vector<int32_t> generated(sequence + ids.size(), sequence + ids.size() + 64);
  WriteBinary(directory / "generated-ids.i32le", generated.data(), generated.size() * sizeof(int32_t));
  std::ofstream tokens(directory / "generated-ids.json");
  Require(tokens.is_open(), "Cannot save generated token IDs");
  tokens << "[";
  for (size_t i = 0; i < generated.size(); ++i) tokens << (i ? "," : "") << generated[i];
  tokens << "]\n";
  tokens.close();
  Require(!tokens.fail(), "Cannot finish saving generated token IDs");
}

inline void SaveState(const fs::path& directory, int context, size_t count, bool phase_b) {
  std::string state = "{\"nominal_context\":" + std::to_string(context) +
                      ",\"input_tokens\":" + std::to_string(count) +
                      ",\"sequence_tokens\":" + std::to_string(count + 64) +
                      ",\"generated_tokens\":64,\"max_length\":" + std::to_string(count + 64) +
                      ",\"chunk_size\":0,\"shrink_calls\":0";
  if (phase_b) state += ",\"diagnostic_logits_calls\":0,\"model_alive_at_cleanup_snapshot\":true";
  Write(directory / "capture-state.json", state + "}\n");
}
