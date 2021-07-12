// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "embed_layer_norm_test_vectors.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(EmbedLayerNormBiasGelu, ShouldWork) {
  //
  // TODO(kreeger): write me!
  //

  // clang-format off
  const std::vector<float> sln1_input_0 = {
    0.877292f, 0.686702f, 0.26983f, 0.410746f,
    0.299725f, 0.314675f, 0.233526f, 0.0724602f,
    0.56409f, 0.30251f, 0.94781f, 0.706121f,
    0.0640292f, 0.301493f, 0.964735f, 0.636928f,
    0.608366f, 0.622913f, 0.395354f, 0.40979f,
    0.427133f, 0.920674f, 0.428131f, 0.626119f,
    0.73837f, 0.480352f, 0.421635f, 0.287868f,
    0.451956f, 0.167709f, 0.235804f, 0.873725f,
  };

  const std::vector<float> sln1_input_1 = {
    0.548171f, 0.993236f, 0.214446f, 0.359704f,
    0.686656f, 0.0831026f, 0.236436f, 0.434125f,
    0.685505f, 0.906413f, 0.547314f, 0.105903f,
    0.689571f, 0.0312862f, 0.696121f, 0.51158f,
    0.0757117f, 0.536431f, 0.68158f, 0.566632f,
    0.381293f, 0.823399f, 0.223761f, 0.861105f,
    0.5068f, 0.693589f, 0.831246f, 0.681481f,
    0.0792981f, 0.234749f, 0.920688f, 0.0283016f,
  };

  const std::vector<float> sln2_input_0 = {
    0.496457f, 0.459715f, 0.673254f, 0.19773f,
    0.984584f, 0.296001f, 0.427254f, 0.870284f,
    0.351943f, 0.226543f, 0.918268f, 0.799669f,
    0.488443f, 0.20992f, 0.0499725f, 0.508568f,
    0.885923f, 0.351155f, 0.586546f, 0.701277f,
    0.812556f, 0.95393f, 0.226148f, 0.642422f,
    0.89366f, 0.933254f, 0.179351f, 0.477409f,
    0.0882888f, 0.429939f, 0.854087f, 0.184111f,
  };

  const std::vector<float> output = {
    0.931759f, 0.338765f, -0.131476f, 1.25103f,
    0.998018f, 0.271481f, -0.0677575f, 1.2027f,
    0.609153f, 0.311766f, -0.0742768f, 1.39308f,
    0.912295f, 0.453042f, -0.203592f, 1.14993f,
    0.973502f, 0.484089f, -0.217615f, 1.05523f,
    0.822251f, 0.49487f, -0.21649f, 1.13241f,
    0.869146f, 0.611431f, -0.232114f, 0.797104f,
    0.148023f, 0.656236f, -0.182884f, 1.04682f,
  };
  // clang-format on
}

}  // namespace test
}  // namespace onnxruntime

