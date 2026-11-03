#include <iostream>

#include "onnxruntime_c_api.h"

int main(int argc, char** argv) {
  const auto* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (api == nullptr || argc != 2) {
    std::cerr << "Expected a compatible runtime and one contrib operator name\n";
    return 2;
  }
  OrtEnv* env = nullptr;
  if (auto* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "schema-probe", &env)) {
    std::cerr << api->GetErrorMessage(status) << '\n';
    api->ReleaseStatus(status);
    return 1;
  }
  const auto* ep_api = api->GetEpApi();
  OrtOpSchema* schema = nullptr;
  if (auto* status = ep_api->GetOpSchema(argv[1], 1, "com.microsoft", &schema)) {
    std::cerr << api->GetErrorMessage(status) << '\n';
    api->ReleaseStatus(status);
    api->ReleaseEnv(env);
    return 1;
  }
  const bool present = schema != nullptr;
  std::cout << argv[1] << ": " << (present ? "schema present" : "schema absent") << '\n';
  if (schema) {
    ep_api->ReleaseOpSchema(schema);
  }
  api->ReleaseEnv(env);
  return present ? 0 : 3;
}
