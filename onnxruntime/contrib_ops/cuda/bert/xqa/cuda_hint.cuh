/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "platform.h"

#if IS_IN_IDE_PARSER

#ifndef __CUDACC__
#define __CUDACC__ 1
#endif

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 900
#endif

#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ 12
#endif
#ifndef __CUDACC_VER_MINOR__
#define __CUDACC_VER_MINOR__ 9
#endif

#if __CUDA_ARCH__ == 900
#ifndef __CUDA_ARCH_FEAT_SM90_ALL
#define __CUDA_ARCH_FEAT_SM90_ALL
#endif
#endif

#endif
