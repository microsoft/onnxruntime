// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import  "OrtSession.h"

#include <vector>
#include <chrono>
#include <sstream>

#include <core/session/onnxruntime_cxx_api.h>


NS_ASSUME_NONNULL_BEGIN


static std::string run_mobilenet(Ort::Session* session) {
    static const int width_ = 224;
    static const int height_ = 224;
    static const int classes = 1000;

    auto& input_image_ = *(new std::array<float, 3 * width_ * height_>());
    auto& results_ = *(new std::array<float, classes>);

    std::array<int64_t, 4> input_shape_{1, 3, width_, height_};
    std::array<int64_t, 2> output_shape_{1, classes};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

    const char* input_names[] = {"data"};
    const char* output_names[] = {"mobilenetv20_output_flatten0_reshape0"};
    
    // Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();
    session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
    
    // Stop measuring time and calculate the elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    Ort::OrtRelease(input_tensor.release());
    Ort::OrtRelease(output_tensor.release());
    delete &input_image_;
    delete &results_;
    
    std::ostringstream output;
    output << "Total time: " << static_cast<double>(elapsed.count() * 1e-3) << std::endl;
    return output.str();
}

static std::string run_nlp(Ort::Session* session) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto input_shape = std::vector<int64_t>({5, 6});
    auto input = new std::array<float, 5 * 6>();
    std::generate(input->begin(), input->end(), []{return std::rand() % 109;});
    
    auto output_shape = std::vector<int64_t>({5, 26});
    auto result = new std::array<float, 5 * 26>();
    
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input->data(), input->size(), input_shape.data(), input_shape.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, result->data(), result->size(), output_shape.data(), output_shape.size());

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"dense_1"};
    
    // Start measuring time
    auto begin = std::chrono::high_resolution_clock::now();
    session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
    
    // Stop measuring time and calculate the elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    delete input;
    delete result;
    
    std::ostringstream output;
    output << "Total time: " << static_cast<double>(elapsed.count() * 1e-3) << std::endl;
    return output.str();
}

@interface OrtMobileSession ()

@property(nonatomic, nullable) Ort::Session* pOrtApiSession;

@property(nonatomic, nullable) Ort::Value* input_tensor;

@property(nonatomic, nullable) Ort::Value* output_tensor;

@end


@implementation OrtMobileSession

#pragma mark - NSObject

- (void)dealloc {
    if (_pOrtApiSession != nullptr) {
        delete _pOrtApiSession;
        _pOrtApiSession = nullptr;
    }
}

#pragma mark - Public

static std::unique_ptr<Ort::Env> ort_env;

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {

    self = [super init];
    if (_pOrtApiSession != nullptr) {
        delete _pOrtApiSession;
        _pOrtApiSession = nullptr;
    }
    ort_env.reset();
    
    ort_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default"));
    Ort::SessionOptions so(nullptr);
    const char* model_path = [modelPath UTF8String];
    _pOrtApiSession = new Ort::Session(*ort_env, model_path, so);
    return self;
}

- (NSString* )run:(nonnull NSMutableData *)buff mname:(NSString*)mname error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    std::string sinfo;
    if ([mname isEqualToString:@"mobilenet"]) {
        sinfo = run_mobilenet(_pOrtApiSession);
    }
    else if ([mname isEqualToString:@"nlp"]) {
        sinfo = run_nlp(_pOrtApiSession);
    }
    
    NSString *resultMsg = [NSString stringWithCString:sinfo.c_str()
                                                encoding:[NSString defaultCStringEncoding]];
    return resultMsg;
}

@end

NS_ASSUME_NONNULL_END
