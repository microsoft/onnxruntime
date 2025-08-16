#include "SessionUtils.h"
#include "JsiUtils.h"
#include <cpu_provider_factory.h>
#include <jsi/jsi.h>
#include <onnxruntime_cxx_api.h>
#ifdef USE_NNAPI
#include <nnapi_provider_factory.h>
#endif
#ifdef USE_COREML
#include <coreml_provider_factory.h>
#endif

// Note: Using below syntax for including ort c api and ort extensions headers to resolve a compiling error happened
// in an expo react native ios app when ort extensions enabled (a redefinition error of multiple object types defined
// within ORT C API header). It's an edge case that compiler allows both ort c api headers to be included when #include
// syntax doesn't match. For the case when extensions not enabled, it still requires a onnxruntime prefix directory for
// searching paths. Also in general, it's a convention to use #include for C/C++ headers rather then #import. See:
// https://google.github.io/styleguide/objcguide.html#import-and-include
// https://microsoft.github.io/objc-guide/Headers/ImportAndInclude.html
#if defined(ORT_ENABLE_EXTENSIONS) && defined(__APPLE__)
#include <onnxruntime_extensions.h>
#endif

using namespace facebook::jsi;

namespace onnxruntimejsi {

const std::vector<const char*> supportedBackends = {
    "cpu",
    "xnnpack",
#ifdef USE_COREML
    "coreml",
#endif
#ifdef USE_NNAPI
    "nnapi",
#endif
#ifdef USE_QNN
    "qnn",
#endif
};

class ExtendedSessionOptions : public Ort::SessionOptions {
 public:
  ExtendedSessionOptions() = default;

  void AppendExecutionProvider_CPU(int use_arena) {
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CPU(this->p_, use_arena));
  }

  void AddFreeDimensionOverrideByName(const char* name, int64_t value) {
    Ort::ThrowOnError(
        Ort::GetApi().AddFreeDimensionOverrideByName(this->p_, name, value));
  }
#ifdef USE_NNAPI
  void AppendExecutionProvider_Nnapi(uint32_t nnapi_flags) {
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_Nnapi(this->p_, nnapi_flags));
  }
#endif
#ifdef USE_COREML
  void AppendExecutionProvider_CoreML(int flags) {
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CoreML(this->p_, flags));
  }
#endif
};

void parseSessionOptions(Runtime& runtime, const Value& optionsValue,
                         Ort::SessionOptions& sessionOptions) {
  if (!optionsValue.isObject())
    return;

  auto options = optionsValue.asObject(runtime);

  try {
#ifdef ORT_ENABLE_EXTENSIONS
    // ortExtLibPath
    if (options.hasProperty(runtime, "ortExtLibPath")) {
#ifdef __APPLE__
      Ort::ThrowOnError(RegisterCustomOps(sessionOptions, OrtGetApiBase()));
#endif
#ifdef __ANDROID__
      auto prop = options.getProperty(runtime, "ortExtLibPath");
      if (prop.isString()) {
        std::string libraryPath = prop.asString(runtime).utf8(runtime);
        sessionOptions.RegisterCustomOpsLibrary(libraryPath.c_str());
      }
#endif
    }
#endif

    // intraOpNumThreads
    if (options.hasProperty(runtime, "intraOpNumThreads")) {
      auto prop = options.getProperty(runtime, "intraOpNumThreads");
      if (prop.isNumber()) {
        int numThreads = static_cast<int>(prop.asNumber());
        if (numThreads > 0) {
          sessionOptions.SetIntraOpNumThreads(numThreads);
        }
      }
    }

    // interOpNumThreads
    if (options.hasProperty(runtime, "interOpNumThreads")) {
      auto prop = options.getProperty(runtime, "interOpNumThreads");
      if (prop.isNumber()) {
        int numThreads = static_cast<int>(prop.asNumber());
        if (numThreads > 0) {
          sessionOptions.SetInterOpNumThreads(numThreads);
        }
      }
    }

    // freeDimensionOverrides
    if (options.hasProperty(runtime, "freeDimensionOverrides")) {
      auto prop = options.getProperty(runtime, "freeDimensionOverrides");
      if (prop.isObject()) {
        auto overrides = prop.asObject(runtime);
        forEach(runtime, overrides,
                [&](const std::string& key, const Value& value, size_t index) {
                  reinterpret_cast<ExtendedSessionOptions&>(sessionOptions)
                      .AddFreeDimensionOverrideByName(
                          key.c_str(), static_cast<int64_t>(value.asNumber()));
                });
      }
    }

    // graphOptimizationLevel
    if (options.hasProperty(runtime, "graphOptimizationLevel")) {
      auto prop = options.getProperty(runtime, "graphOptimizationLevel");
      if (prop.isString()) {
        std::string level = prop.asString(runtime).utf8(runtime);
        if (level == "disabled") {
          sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
        } else if (level == "basic") {
          sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        } else if (level == "extended") {
          sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        } else if (level == "all") {
          sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        }
      }
    }

    // enableCpuMemArena
    if (options.hasProperty(runtime, "enableCpuMemArena")) {
      auto prop = options.getProperty(runtime, "enableCpuMemArena");
      if (prop.isBool()) {
        if (prop.asBool()) {
          sessionOptions.EnableCpuMemArena();
        } else {
          sessionOptions.DisableCpuMemArena();
        }
      }
    }

    // enableMemPattern
    if (options.hasProperty(runtime, "enableMemPattern")) {
      auto prop = options.getProperty(runtime, "enableMemPattern");
      if (prop.isBool()) {
        if (prop.asBool()) {
          sessionOptions.EnableMemPattern();
        } else {
          sessionOptions.DisableMemPattern();
        }
      }
    }

    // executionMode
    if (options.hasProperty(runtime, "executionMode")) {
      auto prop = options.getProperty(runtime, "executionMode");
      if (prop.isString()) {
        std::string mode = prop.asString(runtime).utf8(runtime);
        if (mode == "sequential") {
          sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
        } else if (mode == "parallel") {
          sessionOptions.SetExecutionMode(ORT_PARALLEL);
        }
      }
    }

    // optimizedModelFilePath
    if (options.hasProperty(runtime, "optimizedModelFilePath")) {
      auto prop = options.getProperty(runtime, "optimizedModelFilePath");
      if (prop.isString()) {
        std::string path = prop.asString(runtime).utf8(runtime);
        sessionOptions.SetOptimizedModelFilePath(path.c_str());
      }
    }

    // enableProfiling
    if (options.hasProperty(runtime, "enableProfiling")) {
      auto prop = options.getProperty(runtime, "enableProfiling");
      if (prop.isBool() && prop.asBool()) {
        sessionOptions.EnableProfiling("onnxruntime_profile_");
      }
    }

    // profileFilePrefix
    if (options.hasProperty(runtime, "profileFilePrefix")) {
      auto enableProfilingProp =
          options.getProperty(runtime, "enableProfiling");
      if (enableProfilingProp.isBool() && enableProfilingProp.asBool()) {
        auto prop = options.getProperty(runtime, "profileFilePrefix");
        if (prop.isString()) {
          std::string prefix = prop.asString(runtime).utf8(runtime);
          sessionOptions.EnableProfiling(prefix.c_str());
        }
      }
    }

    // logId
    if (options.hasProperty(runtime, "logId")) {
      auto prop = options.getProperty(runtime, "logId");
      if (prop.isString()) {
        std::string logId = prop.asString(runtime).utf8(runtime);
        sessionOptions.SetLogId(logId.c_str());
      }
    }

    // logSeverityLevel
    if (options.hasProperty(runtime, "logSeverityLevel")) {
      auto prop = options.getProperty(runtime, "logSeverityLevel");
      if (prop.isNumber()) {
        int level = static_cast<int>(prop.asNumber());
        if (level >= 0 && level <= 4) {
          sessionOptions.SetLogSeverityLevel(level);
        }
      }
    }

    // externalData
    if (options.hasProperty(runtime, "externalData")) {
      auto prop =
          options.getProperty(runtime, "externalData").asObject(runtime);
      if (prop.isArray(runtime)) {
        auto externalDataArray = prop.asArray(runtime);
        std::vector<std::string> paths;
        std::vector<char*> buffs;
        std::vector<size_t> sizes;
        forEach(
            runtime, externalDataArray, [&](const Value& value, size_t index) {
              if (value.isObject()) {
                auto externalDataObject = value.asObject(runtime);
                if (externalDataObject.hasProperty(runtime, "path")) {
                  auto pathValue =
                      externalDataObject.getProperty(runtime, "path");
                  if (pathValue.isString()) {
                    paths.push_back(pathValue.asString(runtime).utf8(runtime));
                  }
                }
                if (externalDataObject.hasProperty(runtime, "data")) {
                  auto dataValue =
                      externalDataObject.getProperty(runtime, "data")
                          .asObject(runtime);
                  if (isTypedArray(runtime, dataValue)) {
                    auto arrayBuffer = dataValue.getProperty(runtime, "buffer")
                                           .asObject(runtime)
                                           .getArrayBuffer(runtime);
                    buffs.push_back(
                        reinterpret_cast<char*>(arrayBuffer.data(runtime)));
                    sizes.push_back(arrayBuffer.size(runtime));
                  }
                }
              }
            });
        sessionOptions.AddExternalInitializersFromFilesInMemory(paths, buffs,
                                                                sizes);
      }
    }

    // executionProviders
    if (options.hasProperty(runtime, "executionProviders")) {
      auto prop = options.getProperty(runtime, "executionProviders");
      if (prop.isObject() && prop.asObject(runtime).isArray(runtime)) {
        auto providers = prop.asObject(runtime).asArray(runtime);
        forEach(runtime, providers, [&](const Value& epValue, size_t index) {
          std::string epName;
          std::unique_ptr<Object> providerObj;
          if (epValue.isString()) {
            epName = epValue.asString(runtime).utf8(runtime);
          } else if (epValue.isObject()) {
            providerObj = std::make_unique<Object>(epValue.asObject(runtime));
            epName = providerObj->getProperty(runtime, "name")
                         .asString(runtime)
                         .utf8(runtime);
          }

          // Apply execution providers
          if (epName == "cpu") {
            int use_arena = 0;
            if (providerObj && providerObj->hasProperty(runtime, "useArena")) {
              auto useArena = providerObj->getProperty(runtime, "useArena");
              if (useArena.isBool() && useArena.asBool()) {
                use_arena = 1;
              }
            }
            reinterpret_cast<ExtendedSessionOptions&>(sessionOptions)
                .AppendExecutionProvider_CPU(use_arena);
          } else if (epName == "xnnpack") {
            sessionOptions.AppendExecutionProvider("XNNPACK");
          }
#ifdef USE_COREML
          else if (epName == "coreml") {
            int flags = 0;
            if (providerObj &&
                providerObj->hasProperty(runtime, "coreMlFlags")) {
              auto flagsValue =
                  providerObj->getProperty(runtime, "coreMlFlags");
              if (flagsValue.isNumber()) {
                flags = static_cast<int>(flagsValue.asNumber());
              }
            }
            reinterpret_cast<ExtendedSessionOptions&>(sessionOptions)
                .AppendExecutionProvider_CoreML(flags);
          }
#endif
#ifdef USE_NNAPI
          else if (epName == "nnapi") {
            uint32_t nnapi_flags = 0;
            if (providerObj && providerObj->hasProperty(runtime, "useFP16")) {
              auto useFP16 = providerObj->getProperty(runtime, "useFP16");
              if (useFP16.isBool() && useFP16.asBool()) {
                nnapi_flags |= NNAPI_FLAG_USE_FP16;
              }
            }
            if (providerObj && providerObj->hasProperty(runtime, "useNCHW")) {
              auto useNCHW = providerObj->getProperty(runtime, "useNCHW");
              if (useNCHW.isBool() && useNCHW.asBool()) {
                nnapi_flags |= NNAPI_FLAG_USE_NCHW;
              }
            }
            if (providerObj &&
                providerObj->hasProperty(runtime, "cpuDisabled")) {
              auto cpuDisabled =
                  providerObj->getProperty(runtime, "cpuDisabled");
              if (cpuDisabled.isBool() && cpuDisabled.asBool()) {
                nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
              }
            }
            if (providerObj && providerObj->hasProperty(runtime, "cpuOnly")) {
              auto cpuOnly = providerObj->getProperty(runtime, "cpuOnly");
              if (cpuOnly.isBool() && cpuOnly.asBool()) {
                nnapi_flags |= NNAPI_FLAG_CPU_ONLY;
              }
            }
            reinterpret_cast<ExtendedSessionOptions&>(sessionOptions)
                .AppendExecutionProvider_Nnapi(nnapi_flags);
          }
#endif
#ifdef USE_QNN
          else if (epName == "qnn") {
            std::unordered_map<std::string, std::string> options;
            if (providerObj &&
                providerObj->hasProperty(runtime, "backendType")) {
              options["backendType"] =
                  providerObj->getProperty(runtime, "backendType")
                      .asString(runtime)
                      .utf8(runtime);
            }
            if (providerObj &&
                providerObj->hasProperty(runtime, "backendPath")) {
              options["backendPath"] =
                  providerObj->getProperty(runtime, "backendPath")
                      .asString(runtime)
                      .utf8(runtime);
            }
            if (providerObj &&
                providerObj->hasProperty(runtime, "enableFp16Precision")) {
              auto enableFp16Precision =
                  providerObj->getProperty(runtime, "enableFp16Precision");
              if (enableFp16Precision.isBool() &&
                  enableFp16Precision.asBool()) {
                options["enableFp16Precision"] = "1";
              } else {
                options["enableFp16Precision"] = "0";
              }
            }
            sessionOptions.AppendExecutionProvider("QNN", options);
          }
#endif
          else {
            throw JSError(runtime, "Unsupported execution provider: " + epName);
          }
        });
      }
    }
  } catch (const JSError& e) {
    throw e;
  } catch (const std::exception& e) {
    throw JSError(runtime,
                  "Failed to parse session options: " + std::string(e.what()));
  }
}

void parseRunOptions(Runtime& runtime, const Value& optionsValue,
                     Ort::RunOptions& runOptions) {
  if (!optionsValue.isObject())
    return;

  auto options = optionsValue.asObject(runtime);

  try {
    // tag
    if (options.hasProperty(runtime, "tag")) {
      auto prop = options.getProperty(runtime, "tag");
      if (prop.isString()) {
        std::string tag = prop.asString(runtime).utf8(runtime);
        runOptions.SetRunTag(tag.c_str());
      }
    }

    // logSeverityLevel
    if (options.hasProperty(runtime, "logSeverityLevel")) {
      auto prop = options.getProperty(runtime, "logSeverityLevel");
      if (prop.isNumber()) {
        int level = static_cast<int>(prop.asNumber());
        if (level >= 0 && level <= 4) {
          runOptions.SetRunLogSeverityLevel(level);
        }
      }
    }

    // logVerbosityLevel
    if (options.hasProperty(runtime, "logVerbosityLevel")) {
      auto prop = options.getProperty(runtime, "logVerbosityLevel");
      if (prop.isNumber()) {
        int level = static_cast<int>(prop.asNumber());
        if (level >= 0) {
          runOptions.SetRunLogVerbosityLevel(level);
        }
      }
    }

    // terminate
    if (options.hasProperty(runtime, "terminate")) {
      auto prop = options.getProperty(runtime, "terminate");
      if (prop.isBool() && prop.asBool()) {
        runOptions.SetTerminate();
      }
    }

  } catch (const std::exception& e) {
    throw JSError(runtime,
                  "Failed to parse run options: " + std::string(e.what()));
  }
}

}  // namespace onnxruntimejsi
