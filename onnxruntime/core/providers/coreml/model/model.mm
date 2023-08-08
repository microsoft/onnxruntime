// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/model/model.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/providers/coreml/shape_utils.h"

// force the linker to create a dependency on the CoreML framework so that in MAUI usage we don't need
// to manually do this
asm(".linker_option \"-framework\", \"CoreML\"");

using namespace onnxruntime;
using namespace onnxruntime::coreml;

namespace {
/**
 * Computes the static output shape used to allocate the output tensor.
 * `inferred_shape` is the inferred shape known at model compile time. It may contain dynamic dimensions (-1).
 * `coreml_static_shape` is the static output shape of the CoreML MLMultiArray output. It must NOT contain dynamic
 * dimensions.
 */
InlinedVector<int64_t> GetStaticOutputShape(gsl::span<const int64_t> inferred_shape,
                                            gsl::span<const int64_t> coreml_static_shape,
                                            const logging::Logger& logger) {
  ORT_ENFORCE(IsStaticShape(coreml_static_shape),
              "CoreML output shape (", Shape2String(coreml_static_shape), ") is not static.");

  // return early if the shapes match
  if (std::equal(inferred_shape.begin(), inferred_shape.end(),
                 coreml_static_shape.begin(), coreml_static_shape.end())) {
    return InlinedVector<int64_t>(coreml_static_shape.begin(), coreml_static_shape.end());
  }

  // Special CoreML behavior notes:
  // - Sometimes the CoreML output shape has extra leading ones.

  ORT_ENFORCE(inferred_shape.size() <= coreml_static_shape.size(),
              "CoreML static output shape (", Shape2String(coreml_static_shape),
              ") has fewer elements than the inferred shape (", Shape2String(inferred_shape), ").");

  // if coreml_static_shape has more elements, we expect them to be leading ones
  const size_t num_leading_dimensions = coreml_static_shape.size() - inferred_shape.size();
  const auto coreml_static_shape_common_begin = coreml_static_shape.begin() + num_leading_dimensions;

  if (num_leading_dimensions > 0) {
    const bool has_only_leading_ones =
        std::all_of(coreml_static_shape.begin(), coreml_static_shape_common_begin,
                    [](int64_t dim) { return dim == 1; });
    ORT_ENFORCE(has_only_leading_ones, "CoreML static output shape (", Shape2String(coreml_static_shape),
                ") has leading dimensions with value other than 1.");
  }

  InlinedVector<int64_t> static_shape{};
  static_shape.reserve(inferred_shape.size());
  std::transform(inferred_shape.begin(), inferred_shape.end(),
                 coreml_static_shape_common_begin,
                 std::back_inserter(static_shape),
                 [&](int64_t inferred_dim, int64_t coreml_static_dim) {
                   ORT_ENFORCE(inferred_dim == -1 || inferred_dim == coreml_static_dim,
                               "CoreML static output shape (", Shape2String(coreml_static_shape),
                               ") and inferred shape (", Shape2String(inferred_shape),
                               ") have an inconsistent static dimensions (", coreml_static_dim, " vs. ",
                               inferred_dim, ").");

                   return inferred_dim != -1 ? inferred_dim : coreml_static_dim;
                 });

  // Ideally, the CoreML static shape would match the inferred shape exactly, apart from the former providing values
  // for -1's in the latter. For now, this is not the case so it is probably worth logging them.
  LOGS(logger, VERBOSE) << "CoreML static output shape: " << Shape2String(coreml_static_shape)
                        << ", inferred shape: " << Shape2String(inferred_shape)
                        << ", resulting static output shape: " << Shape2String(static_shape);
  return static_shape;
}
}  // namespace

NS_ASSUME_NONNULL_BEGIN

// Model input for a CoreML model
// All the input onnx tensors values will be converted to MLMultiArray(s)
@interface OnnxTensorFeatureProvider : NSObject <MLFeatureProvider> {
  const std::unordered_map<std::string, OnnxTensorData>* inputs_;
  NSSet* featureNames_;
  const logging::Logger* logger_;
}

- (instancetype)initWithInputs:(const std::unordered_map<std::string, OnnxTensorData>&)inputs
                        logger:(const logging::Logger&)logger;
- (nullable MLFeatureValue*)featureValueForName:(NSString*)featureName API_AVAILABLE_OS_VERSIONS;
- (NSSet<NSString*>*)featureNames;

@end

// Execution for a CoreML model, it performs
// 1. Compile the model by given path for execution
// 2. Predict using given OnnxTensorFeatureProvider input and copy the output data back ORT
// 3. The compiled model will be removed in dealloc or removed using cleanup function
@interface CoreMLExecution : NSObject {
  NSString* coreml_model_path_;
  NSString* compiled_model_path_;
  const logging::Logger* logger_;
  uint32_t coreml_flags_;
}

- (instancetype)initWithPath:(const std::string&)path
                      logger:(const logging::Logger&)logger
                coreml_flags:(uint32_t)coreml_flags;
- (void)cleanup;
- (void)dealloc;
- (Status)loadModel API_AVAILABLE_OS_VERSIONS;
- (Status)predict:(const std::unordered_map<std::string, OnnxTensorData>&)inputs
                  outputs:(const std::unordered_map<std::string, OnnxTensorInfo>&)outputs
    getOutputTensorDataFn:(const GetOutputTensorMutableRawDataFn&)
                              get_output_tensor_mutable_raw_data_fn
    API_AVAILABLE_OS_VERSIONS;

@property MLModel* model API_AVAILABLE_OS_VERSIONS;

@end

@implementation OnnxTensorFeatureProvider

- (instancetype)initWithInputs:(const std::unordered_map<std::string, OnnxTensorData>&)inputs
                        logger:(const logging::Logger&)logger {
  if (self = [super init]) {
    inputs_ = &inputs;
    logger_ = &logger;
  }
  return self;
}

- (NSSet<NSString*>*)featureNames {
  if (featureNames_ == nil) {
    NSMutableArray* names = [[NSMutableArray alloc] init];
    for (const auto& input : *inputs_) {
      NSString* inputName = [NSString stringWithCString:input.first.c_str()
                                               encoding:[NSString defaultCStringEncoding]];
      NSAssert(inputName != nil, @"inputName must not be nil");
      [names addObject:inputName];
    }

    featureNames_ = [NSSet setWithArray:names];
  }

  return featureNames_;
}

- (nullable MLFeatureValue*)featureValueForName:(NSString*)featureName {
  auto it = inputs_->find([featureName cStringUsingEncoding:NSUTF8StringEncoding]);
  if (it != inputs_->end()) {
    auto& input = it->second;
    NSMutableArray* shape = [[NSMutableArray alloc] init];
    for (const auto dim : input.tensor_info.shape) {
      [shape addObject:[NSNumber numberWithLongLong:dim]];
    }

    NSMutableArray* strides = [[NSMutableArray alloc] init];
    int64_t stride = 1;
    for (int i = static_cast<int>(input.tensor_info.shape.size()) - 1; i >= 0; i--) {
      [strides insertObject:[NSNumber numberWithLongLong:stride]
                    atIndex:0];

      stride *= input.tensor_info.shape[i];
    }

    MLMultiArrayDataType data_type = MLMultiArrayDataTypeFloat32;
    if (input.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      LOGS(*logger_, ERROR) << "Input data type is not float, actual type: "
                            << input.tensor_info.data_type;
      return nil;
    }

    NSError* error = nil;
    MLMultiArray* mlArray = [[MLMultiArray alloc] initWithDataPointer:input.buffer
                                                                shape:shape
                                                             dataType:data_type
                                                              strides:strides
                                                          deallocator:^(void* /* bytes */) {
                                                          }
                                                                error:&error];
    if (error != nil) {
      LOGS(*logger_, ERROR) << "Failed to create MLMultiArray for feature: " << [featureName UTF8String]
                            << ", error: " << [[error localizedDescription] UTF8String];
      return nil;
    }

    NSAssert(mlArray != nil, @"mlArray must not be nil");
    auto* mlFeatureValue = [MLFeatureValue featureValueWithMultiArray:mlArray];
    return mlFeatureValue;
  }

  return nil;
}

@end

@implementation CoreMLExecution

- (instancetype)initWithPath:(const std::string&)path
                      logger:(const logging::Logger&)logger
                coreml_flags:(uint32_t)coreml_flags {
  if (self = [super init]) {
    coreml_model_path_ = [NSString stringWithUTF8String:path.c_str()];
    logger_ = &logger;
    coreml_flags_ = coreml_flags;
  }
  return self;
}

- (void)cleanup {
  NSError* error = nil;
  if (compiled_model_path_ != nil) {
    [[NSFileManager defaultManager] removeItemAtPath:compiled_model_path_ error:&error];
    if (error != nil) {
      LOGS(*logger_, ERROR) << "Failed cleaning up the compiled model: " << [compiled_model_path_ UTF8String]
                            << ", error message: " << [[error localizedDescription] UTF8String];
    }
    compiled_model_path_ = nil;
  }

  if (coreml_model_path_ != nil) {
    error = nil;
    [[NSFileManager defaultManager] removeItemAtPath:coreml_model_path_ error:&error];
    if (error != nil) {
      LOGS(*logger_, ERROR) << "Failed cleaning up the coreml model: " << [coreml_model_path_ UTF8String]
                            << ", error message: " << [[error localizedDescription] UTF8String];
    }
    coreml_model_path_ = nil;
  }
}

- (void)dealloc {
  [self cleanup];
}

- (Status)loadModel {
  NSError* error = nil;
  NSURL* modelUrl = [NSURL URLWithString:coreml_model_path_];
  NSAssert(modelUrl != nil, @"modelUrl must not be nil");
  NSURL* compileUrl = [MLModel compileModelAtURL:modelUrl error:&error];

  if (error != nil) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error compiling model ",
                           [[error localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding]);
  }

  compiled_model_path_ = [compileUrl path];

  MLModelConfiguration* config = [MLModelConfiguration alloc];
  config.computeUnits = (coreml_flags_ & COREML_FLAG_USE_CPU_ONLY)
                            ? MLComputeUnitsCPUOnly
                            : MLComputeUnitsAll;
  _model = [MLModel modelWithContentsOfURL:compileUrl configuration:config error:&error];

  if (error != NULL) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error Creating MLModel ",
                           [[error localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding]);
  }

  return Status::OK();
}

- (Status)predict:(const std::unordered_map<std::string, OnnxTensorData>&)inputs
                  outputs:(const std::unordered_map<std::string, OnnxTensorInfo>&)outputs
    getOutputTensorDataFn:(const GetOutputTensorMutableRawDataFn&)get_output_tensor_mutable_raw_data_fn {
  Status status = Status::OK();
  ORT_TRY {
    if (_model == nil) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model is not loaded");
    }

    OnnxTensorFeatureProvider* input_feature = [[OnnxTensorFeatureProvider alloc] initWithInputs:inputs
                                                                                          logger:*logger_];

    if (input_feature == nil) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "inputFeature is not initialized");
    }

    MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
    NSError* error = nil;
    id<MLFeatureProvider> output_feature = [_model predictionFromFeatures:input_feature
                                                                  options:options
                                                                    error:&error];

    if (error != nil) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error executing model: ",
                             [[error localizedDescription] cStringUsingEncoding:NSUTF8StringEncoding]);
    }

    for (const auto& [output_name, output_tensor_info] : outputs) {
      MLFeatureValue* output_value =
          [output_feature featureValueForName:[NSString stringWithUTF8String:output_name.c_str()]];

      if (output_value == nil) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "output_feature has no value for ", output_name);
      }

      auto* data = [output_value multiArrayValue];

      const auto coreml_static_output_shape = [&]() {
        InlinedVector<int64_t> result;
        result.reserve(data.shape.count);
        for (NSNumber* dim in data.shape) {
          const auto dim_value = dim.longLongValue;
          result.push_back(dim_value);
        }
        return result;
      }();

      const auto static_output_shape = GetStaticOutputShape(output_tensor_info.shape, coreml_static_output_shape,
                                                            *logger_);

      void* output_buffer = get_output_tensor_mutable_raw_data_fn(output_name, output_tensor_info.data_type,
                                                                  static_output_shape);

      if (const size_t num_elements = data.count; num_elements > 0) {
        if (const auto shape_size = ShapeSize(static_output_shape);
            shape_size < 0 || num_elements != static_cast<size_t>(shape_size)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "CoreML MLMultiArray count (", num_elements, ") and shape size (", shape_size,
                                 ") do not match");
        }

        const void* model_output_data = data.dataPointer;

        if (model_output_data == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model_output_data has no data for ", output_name);
        }

        const auto onnx_data_type = output_tensor_info.data_type;
        switch (onnx_data_type) {
          case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
            const auto output_data_byte_size = num_elements * sizeof(float);
            memcpy(output_buffer, model_output_data, output_data_byte_size);
            break;
          }
          case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
            const auto output_data_byte_size = num_elements * sizeof(int32_t);
            memcpy(output_buffer, model_output_data, output_data_byte_size);
            break;
          }
          // For this case, since Coreml Spec only uses int32 for model output while onnx provides
          // int64 for model output data type. We are doing a type casting (int32 -> int64) here
          // when copying the model to ORT
          case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
            ORT_RETURN_IF_NOT(data.dataType == MLMultiArrayDataTypeInt32,
                              "CoreML output data type is not MLMultiArrayDataTypeInt32");

            const int32_t* model_output_data_i32 = static_cast<const int32_t*>(model_output_data);
            int64_t* output_tensor_buffer_i64 = static_cast<int64_t*>(output_buffer);
            for (size_t i = 0; i < num_elements; i++) {
              output_tensor_buffer_i64[i] = model_output_data_i32[i];
            }
            break;
          }
          default:
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "Output data type is not supported, actual type: ", onnx_data_type);
        }
      }
    }
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exception: ", e.what());
    });
  }

  return status;
}

@end

NS_ASSUME_NONNULL_END

namespace onnxruntime {
namespace coreml {

// Internal Execution class
// This class will bridge Model (c++) with CoreMLExecution (objective c++)
class Execution {
 public:
  Execution(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags);
  ~Execution(){};

  Status LoadModel();
  Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                 const std::unordered_map<std::string, OnnxTensorInfo>& outputs,
                 const GetOutputTensorMutableRawDataFn& get_output_tensor_mutable_raw_data_fn);

 private:
  bool model_loaded{false};
  CoreMLExecution* execution_;
};

Execution::Execution(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags) {
  @autoreleasepool {
    execution_ = [[CoreMLExecution alloc] initWithPath:path
                                                logger:logger
                                          coreml_flags:coreml_flags];
  }
}

Status Execution::LoadModel() {
  if (model_loaded) {
    return Status::OK();
  }

  if (HAS_VALID_BASE_OS_VERSION) {
    Status status{};
    @autoreleasepool {
      status = [execution_ loadModel];
    }
    model_loaded = status.IsOK();
    return status;
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Execution::LoadModel requires macos 10.15+ or ios 13+ ");
}

Status Execution::Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                          const std::unordered_map<std::string, OnnxTensorInfo>& outputs,
                          const GetOutputTensorMutableRawDataFn& get_output_tensor_mutable_raw_data_fn) {
  ORT_RETURN_IF_NOT(model_loaded, "Execution::Predict requires Execution::LoadModel");

  if (HAS_VALID_BASE_OS_VERSION) {
    @autoreleasepool {
      return [execution_ predict:inputs
                         outputs:outputs
           getOutputTensorDataFn:get_output_tensor_mutable_raw_data_fn];
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Execution::LoadModel requires macos 10.15+ or ios 13+ ");
}

Model::Model(const std::string& path, const logging::Logger& logger, uint32_t coreml_flags)
    : execution_(std::make_unique<Execution>(path, logger, coreml_flags)) {
}

Model::~Model() {}

Status Model::LoadModel() {
  return execution_->LoadModel();
}

Status Model::Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                      const std::unordered_map<std::string, OnnxTensorInfo>& outputs,
                      const GetOutputTensorMutableRawDataFn& get_output_tensor_mutable_raw_data_fn) {
  return execution_->Predict(inputs, outputs, get_output_tensor_mutable_raw_data_fn);
}

bool Model::IsScalarOutput(const std::string& output_name) const {
  return Contains(scalar_outputs_, output_name);
}

bool Model::IsInt64Output(const std::string& output_name) const {
  return Contains(int64_outputs_, output_name);
}

const OnnxTensorInfo* Model::TryGetInputOutputInfo(const std::string& name) const {
  const auto info_it = input_output_info_.find(name);
  return info_it != input_output_info_.end() ? &info_it->second : nullptr;
}

const OnnxTensorInfo& Model::GetInputOutputInfo(const std::string& name) const {
  const auto* info = TryGetInputOutputInfo(name);
  ORT_ENFORCE(info != nullptr, "Failed to get info for input/output: ", name);
  return *info;
}

}  // namespace coreml
}  // namespace onnxruntime
