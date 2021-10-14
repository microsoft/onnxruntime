/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Tools
 **/

#pragma once
#include "contrib_ops/cuda/fastertransformer/utils/common.h"
#include <cuda_runtime.h>

namespace fastertransformer{

/**
 * Pop current cuda device and set new device
 * i_device - device ID to set
 * o_device - device ID to pop
 * ret  - return code (the same as cudaError_t)
 */

inline cudaError_t get_set_device(int i_device, int* o_device = NULL){
  int current_dev_id = 0;
  cudaError_t err = cudaSuccess;

  if (o_device != NULL) {
    err = cudaGetDevice(&current_dev_id);
    if (err != cudaSuccess)
      return err;
    if (current_dev_id == i_device){
      *o_device = i_device;
    }
    else{
      err = cudaSetDevice(i_device);
      if (err != cudaSuccess) {
        return err;
      }
      *o_device = current_dev_id;
    }
  }
  else{
    err = cudaSetDevice(i_device);
    if (err != cudaSuccess) {
      return err;
    }
  }

  return cudaSuccess;
}

}
