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
#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

template<typename T>
void LaunchTopK(
    const T* input,
    int batch_size,
    int num_beams,
    int vocab_size,
    int K,
    T* output_values,
    int32_t* output_indices,
    T* output_values_tmp,
    int32_t* output_indices_tmp,
    cudaStream_t stream);

}
}
}