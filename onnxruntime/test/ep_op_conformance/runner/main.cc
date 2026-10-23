#include "case_io.h"

#include <chrono>
#include <iostream>
#include <map>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ep_op_conformance = onnxruntime::test::ep_op_conformance;
using ep_op_conformance::Json;
namespace fs = std::filesystem;

namespace {

class Runtime {
 public:
  explicit Runtime(const fs::path& path) {
#ifdef _WIN32
    module_ = LoadLibraryExW(fs::absolute(path).c_str(), nullptr,
                             LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    auto get_api_base = module_ ? reinterpret_cast<decltype(&OrtGetApiBase)>(GetProcAddress(module_, "OrtGetApiBase"))
                                : nullptr;
#else
    module_ = dlopen(fs::absolute(path).c_str(), RTLD_NOW | RTLD_LOCAL);
    auto get_api_base = module_ ? reinterpret_cast<decltype(&OrtGetApiBase)>(dlsym(module_, "OrtGetApiBase")) : nullptr;
#endif
    if (!get_api_base) {
      throw std::runtime_error("Cannot load ORT runtime: " + path.string());
    }

    const auto* base = get_api_base();
    const auto* api = base->GetApi(24);
    if (!api) {
      throw std::runtime_error("ORT runtime must support API version 24");
    }

    Ort::InitApi(api);
    version = base->GetVersionString();
  }

  ~Runtime() {
#ifdef _WIN32
    if (module_) {
      FreeLibrary(module_);
    }
#else
    if (module_) {
      dlclose(module_);
    }
#endif
  }

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  auto Symbol(const char* name) const {
#ifdef _WIN32
    return GetProcAddress(module_, name);
#else
    return dlsym(module_, name);
#endif
  }

  std::string version;

 private:
#ifdef _WIN32
  HMODULE module_{};
#else
  void* module_{};
#endif
};

struct Target {
  using DmlAppend = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
  Json profile;
  std::string ep_name;
  std::vector<Ort::ConstEpDevice> devices;
  DmlAppend dml_append{};

  Target(Ort::Env& env, const Runtime& runtime, const fs::path& profile_path)
      : profile(Json::parse(ep_op_conformance::ReadFile(profile_path))), ep_name(profile.at("ep_name").get<std::string>()) {
    ep_op_conformance::CheckFields(profile, {"schema_version", "kind", "ep_name", "options", "library",
                                             "registration_name", "device_index", "skips"});
    if (!profile.at("schema_version").is_number_integer() || profile.at("schema_version") != 1 || ep_name.empty()) {
      throw std::runtime_error("Invalid target profile");
    }

    if (profile.contains("options")) {
      profile.at("options").get<std::unordered_map<std::string, std::string>>();
    }

    const auto kind = profile.at("kind").get<std::string>();
    if (kind == "plugin") {
      ep_op_conformance::CheckFields(profile, {"schema_version", "kind", "ep_name", "options", "skips", "library",
                                               "registration_name", "device_index"});
      if (profile.contains("device_index") &&
          (!profile.at("device_index").is_number_integer() || profile.at("device_index") < 0)) {
        throw std::runtime_error("Invalid plugin device index");
      }

      auto library = fs::path(profile.at("library").get<std::string>());
      if (library.is_relative()) {
        library = profile_path.parent_path() / library;
      }

      env.RegisterExecutionProviderLibrary(profile.at("registration_name").get<std::string>().c_str(),
                                           fs::absolute(library).c_str());
      std::vector<Ort::ConstEpDevice> matching;
      for (const auto& device : env.GetEpDevices()) {
        if (ep_name == device.EpName()) {
          matching.push_back(device);
        }
      }

      const auto index = profile.value("device_index", size_t{0});
      if (index >= matching.size()) {
        throw std::runtime_error("Requested plugin device is unavailable");
      }

      devices.push_back(matching[index]);
    } else if (kind == "builtin") {
      ep_op_conformance::CheckFields(profile, {"schema_version", "kind", "ep_name", "options", "skips"});
      const auto available = Ort::GetAvailableProviders();
      if (std::find(available.begin(), available.end(), ep_name) == available.end()) {
        throw std::runtime_error("Requested non-plugin provider is unavailable: " + ep_name);
      }

      if (ep_name == "CPUExecutionProvider" && profile.contains("options") && !profile.at("options").empty()) {
        throw std::runtime_error("Pilot CPU target does not accept provider options");
      }

      if (ep_name == "DmlExecutionProvider") {
        dml_append = reinterpret_cast<DmlAppend>(runtime.Symbol("OrtSessionOptionsAppendExecutionProvider_DML"));
        if (!dml_append) {
          throw std::runtime_error("Runtime has no public DirectML append entry point");
        }
      }
    } else {
      throw std::runtime_error("Unknown target kind");
    }
  }

  Ort::SessionOptions Options(Ort::Env& env) const {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry("session.record_ep_graph_assignment_info", "1");
    const auto kind = profile.at("kind").get<std::string>();
    if (kind == "builtin" && ep_name == "CPUExecutionProvider") {
      return options;
    }

    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    const auto settings = profile.value("options", std::unordered_map<std::string, std::string>{});
    if (kind == "plugin") {
      options.AppendExecutionProvider_V2(env, devices, settings);
    } else if (ep_name == "CUDAExecutionProvider") {
      Ort::CUDAProviderOptions cuda;
      cuda.Update(settings);
      options.AppendExecutionProvider_CUDA_V2(*cuda);
    } else if (ep_name == "DmlExecutionProvider") {
      for (const auto& [key, value] : settings) {
        static_cast<void>(value);
        if (key != "device_id") {
          throw std::runtime_error("Unsupported DirectML pilot option: " + key);
        }
      }

      const auto device_text = settings.count("device_id") ? settings.at("device_id") : "0";
      size_t consumed{};
      const int device = std::stoi(device_text, &consumed);
      if (consumed != device_text.size() || device < 0) {
        throw std::runtime_error("Invalid DirectML device ID");
      }

      options.DisableMemPattern();
      Ort::ThrowOnError(dml_append(options, device));
    } else {
      options.AppendExecutionProvider(ep_name, settings);
    }
    return options;
  }
};

Json CheckAssignment(const Ort::Session& session, const Json& manifest, const std::string& target) {
  std::set<std::string> required;
  for (const auto& node : manifest.at("required_nodes")) {
    required.insert(node.get<std::string>());
  }

  Json assignments = Json::array();
  for (const auto& subgraph : session.GetEpGraphAssignmentInfo()) {
    const std::string provider = subgraph.GetEpName();
    for (const auto& node : subgraph.GetNodes()) {
      const std::string name = node.GetName();
      assignments.push_back({{"node", name}, {"ep_name", provider}, {"operator", node.GetOperatorType()}});
      if (provider != target) {
        throw std::runtime_error("Unexpected provider assignment: " + name + " -> " + provider);
      }

      required.erase(name);
    }
  }
  if (!required.empty()) {
    throw std::runtime_error("Missing target assignment for required node: " + *required.begin());
  }

  return assignments;
}

void ExecuteDataset(Ort::Session& session, const ep_op_conformance::Dataset& data, const Json& manifest) {
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  for (const auto& input : data.inputs) {
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.values.data()), input.values.size(),
                                                     input.shape.data(), input.shape.size()));
    input_names.push_back(input.name.c_str());
  }
  for (const auto& output : data.outputs) {
    output_names.push_back(output.name.c_str());
  }

  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                             output_names.data(), output_names.size());
  for (size_t index = 0; index < outputs.size(); ++index) {
    const auto& expected = data.outputs[index];
    if (!outputs[index].IsTensor()) {
      throw std::runtime_error("Expected dense tensor output");
    }

    const auto info = outputs[index].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape() != expected.shape) {
      throw std::runtime_error("Output dtype or shape mismatch: " + expected.name);
    }
    ep_op_conformance::CompareFloats(expected.values, outputs[index].GetTensorData<float>(), info.GetElementCount(),
                                     ep_op_conformance::ReadComparison(manifest.at("comparison").at(expected.name)));
  }
}

}  // namespace

int main(int argc, char** argv) {
  Json report{{"schema_version", 1}, {"results", Json::array()}, {"status", "ERROR"}};
  fs::path report_path;
  try {
    if (argc != 9) {
      throw std::runtime_error("Usage: runner --runtime PATH --profile PATH --cases DIRECTORY --report PATH");
    }

    std::map<std::string, fs::path> args;
    for (int index = 1; index < argc; index += 2) {
      const std::string key(argv[index]);
      if (key != "--runtime" && key != "--profile" && key != "--cases" && key != "--report") {
        throw std::runtime_error("Unknown argument: " + key);
      }
      if (!args.emplace(key, fs::path(argv[index + 1])).second) {
        throw std::runtime_error("Duplicate argument");
      }
    }

    report_path = args.at("--report");
    Runtime runtime(args.at("--runtime"));
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ep_op_conformance");
    Target target(env, runtime, args.at("--profile"));
    report["runtime_version"] = runtime.version;
    report["profile"] = target.profile;

    std::vector<fs::path> directories;
    const auto root = args.at("--cases");
    if (fs::exists(root / "case.json")) {
      directories.push_back(root);
    } else {
      for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_directory() && fs::exists(entry.path() / "case.json")) {
          directories.push_back(entry.path());
        }
      }
    }

    std::sort(directories.begin(), directories.end());
    if (directories.empty()) {
      throw std::runtime_error("No cases selected");
    }

    std::vector<ep_op_conformance::Case> cases;
    std::set<std::string> ids;
    for (const auto& directory : directories) {
      auto test = ep_op_conformance::ReadCase(directory);
      if (!ids.insert(test.manifest.at("id").get<std::string>()).second) {
        throw std::runtime_error("Duplicate case ID");
      }

      cases.push_back(std::move(test));
    }

    const auto skips = target.profile.value("skips", std::map<std::string, std::string>{});
    for (const auto& [id, reason] : skips) {
      if (!ids.count(id) || reason.empty()) {
        throw std::runtime_error("Skip must name a selected case and reason");
      }
    }

    size_t failures = 0;
    size_t passes = 0;
    for (const auto& test : cases) {
      const auto id = test.manifest.at("id").get<std::string>();
      Json result{{"id", id}, {"source", test.manifest.at("source")}};
      const auto start = std::chrono::steady_clock::now();
      if (const auto skip = skips.find(id); skip != skips.end()) {
        result["status"] = "SKIP";
        result["reason"] = skip->second;
      } else {
        std::string phase = "session";
        try {
          auto options = target.Options(env);
          Ort::Session session(env, test.model.data(), test.model.size(), options);

          phase = "assignment";
          result["assignment"] = CheckAssignment(session, test.manifest, target.ep_name);

          phase = "execution_or_comparison";
          for (const auto& data : test.datasets) {
            ExecuteDataset(session, data, test.manifest);
          }

          result["status"] = "PASS";
          ++passes;
        } catch (const std::exception& error) {
          result["status"] = "FAIL";
          result["phase"] = phase;
          result["reason"] = error.what();
          ++failures;
        }
      }

      result["duration_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      report["results"].push_back(std::move(result));
    }

    report["status"] = failures || !passes ? "FAIL" : "PASS";
    report["passed"] = passes;
    report["failed"] = failures;
    report["skipped"] = skips.size();
    ep_op_conformance::WriteFile(report_path, report.dump(2));
    std::cout << report.dump(2) << '\n';
    return failures || !passes ? 1 : 0;
  } catch (const std::exception& error) {
    report["error"] = error.what();
    std::cerr << report.dump(2) << '\n';
    if (!report_path.empty()) {
      try {
        ep_op_conformance::WriteFile(report_path, report.dump(2));
      } catch (const std::exception&) {
      }
    }
    return 2;
  }
}