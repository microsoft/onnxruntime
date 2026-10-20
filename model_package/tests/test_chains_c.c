// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_package.h"

#include <stdio.h>
#include <string.h>

static int CheckStatus(ModelPackageStatus* status) {
  if (status == NULL) return 1;
  fprintf(stderr, "%s\n", ModelPackageStatus_Message(status));
  ModelPackageStatus_Release(status);
  return 0;
}

#define CHECK(expression)                               \
  do {                                                  \
    if (!(expression)) {                                \
      fprintf(stderr, "Failed at line %d\n", __LINE__); \
      ModelPackage_Close(package);                      \
      return 1;                                         \
    }                                                   \
    ++checks;                                           \
  } while (0)

int main(void) {
  ModelPackage* package = NULL;
  const char* json = NULL;
  int checks = 0;
  CHECK(CheckStatus(ModelPackage_New(&package)));
  CHECK(ModelPackage_Info(package)->schema_version_major == 1);
  CHECK(ModelPackage_Info(package)->schema_version_minor == 1);
  CHECK(CheckStatus(ModelPackage_GetChainsJson(package, &json)));
  CHECK(strcmp(json, "{}") == 0);
  CHECK(CheckStatus(ModelPackage_SetComponentInline(package, "decoder", "{\"variants\":{}}")));
  CHECK(CheckStatus(ModelPackage_SetChainJson(
      package, "generate", "{\"steps\":[{\"id\":\"prefill\",\"component\":\"decoder\"}]}")));
  CHECK(CheckStatus(ModelPackage_GetChainJson(package, "generate", &json)));
  CHECK(strstr(json, "prefill") != NULL);
  CHECK(CheckStatus(ModelPackage_SetChainsJson(
      package, "{\"other\":{\"steps\":[{\"id\":\"decode\",\"component\":\"decoder\"}]}}")));
  CHECK(CheckStatus(ModelPackage_GetChainsJson(package, &json)));
  CHECK(strstr(json, "decode") != NULL);
  CHECK(CheckStatus(ModelPackage_RemoveChain(package, "other")));
  CHECK(CheckStatus(ModelPackage_GetChainsJson(package, &json)));
  CHECK(strcmp(json, "{}") == 0);
  ModelPackage_Close(package);
  printf("PASS: C11 consumer, %d checks, all five chain APIs.\n", checks);
  return 0;
}
