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

#include "defines.h"
#if SPEC_DEC

struct SpecDecParams {
  uint32_t qSeqLen;
  uint32_t const* qCuSeqLens;  // [nbReq + 1]
  MaskType const* mask;        // [nbReq][qSeqLen][divUp(qSeqLen, 32)] or [qCuSeqLen[nbReq]][divUp(qSeqLen, 32)]
};

#endif
