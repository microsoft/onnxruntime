// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/callback.h"

ORT_API(void, OrtRunCallback, _Frees_ptr_opt_ OrtCallback* f){
  if(f == nullptr) return;
  if(f->f != nullptr) {
    f->f(f->param);
    delete f;
  }
}
