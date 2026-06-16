/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_runtime.h>

inline cudaLaunchConfig_t makeLaunchConfig(
    dim3 const& gridDim, dim3 const& ctaDim, size_t dynShmBytes, cudaStream_t stream, bool usePDL) {
  static cudaLaunchAttribute pdlAttr;
  pdlAttr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdlAttr.val.programmaticStreamSerializationAllowed = (usePDL ? 1 : 0);

  cudaLaunchConfig_t cfg{gridDim, ctaDim, dynShmBytes, stream, &pdlAttr, 1};
  return cfg;
}
