// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_SHA256_H_
#define ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_SHA256_H_

#include <string>

typedef struct {
  unsigned char data[64];
  unsigned int datalen;
  unsigned int bitlen[2];
  unsigned int state[8];
} SHA256_CTX;

void SHA256Transform(SHA256_CTX *ctx, unsigned char data[]);
void SHA256Init(SHA256_CTX *ctx);
void SHA256Update(SHA256_CTX *ctx, unsigned char data[], unsigned int len);
void SHA256Final(SHA256_CTX *ctx, unsigned char hash[]);
std::string SHA256(const char* data, size_t size);

#endif  // ONNXRUNTIME_CORE_PROVIDERS_TVM_HASH_ALG_SHA256_H_
