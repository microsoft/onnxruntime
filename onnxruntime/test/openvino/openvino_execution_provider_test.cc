// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"

namespace onnxruntime {
namespace test {
TEST(OpenVINOExecutionProviderTest, IntegrationTest) {

    OpenVINOExecutionProviderInfo info;
    static const std::string MODEL_URI = "/home/suryasid/Desktop/pcb_hmi_mar5.onnx";
    // static const std::string MODEL_URI = "testdata/alexnet.onnx";
    SessionOptions so;
    so.session_logid = "OpenVINOGetCapability";

    InferenceSession session_object{so, &DefaultLoggingManager()};

    auto opv_provider = std::make_unique<OpenVINOExecutionProvider>(info);
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(opv_provider)).IsOK());
    EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
    EXPECT_TRUE(session_object.Initialize().IsOK());

    RunOptions run_options;
    run_options.run_tag = "one session/one tag";

    std::vector<float> input(150528, 3.0);
    srand(time(nullptr));
    /*
    for(int i = 0; i< 1000; i++){
        //float f = static_cast <float> (rand()/ static_cast <float> (RAND_MAX));
        //input.push_back(f);
      input.push_back(3.0);
    }
    */

    std::cout << "\n API Inputs: \n";
    std::cout << "---------------\n";
    for (int i=0; i<10; i++) {
      std::cout << input[i] << std::endl;
    }
    std::cout << std::endl;

    std::vector<int64_t> dims_x = {1, 3, 224, 224};

    MLValue ml_value;
    CreateMLValue<float>(TestOpenVINOExecutionProvider()->GetAllocator(0,OrtMemTypeDefault), dims_x, input, &ml_value);
    NameMLValMap feeds;
    feeds.insert(std::make_pair("input:0",ml_value));

    std::vector<std::string> output_names;
    output_names.push_back("final_result:0");
    std::vector<MLValue> fetches;
    common::Status st = session_object.Run(run_options,feeds,output_names,&fetches);
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;


    auto& out_tensor = fetches.front().Get<Tensor>();
    const float* output_buffer = (out_tensor.template Data<float>());
    std::cout << "API outputs:" << std::endl;
    std::cout << "------------" << std::endl;
    for(int i=0; i<10; i++) {
    	//std::cout << out_tensor.Get<float>()[i] << std::endl;
    	std::cout << output_buffer[i] << std::endl;
    }
    std::cout << std::endl;
//   ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeDefault)->Info().name, CPU);
}
}  // namespace test
}  // namespace onnxruntime
