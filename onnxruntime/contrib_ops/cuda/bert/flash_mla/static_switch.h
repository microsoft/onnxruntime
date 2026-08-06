/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * reference: https://github.com/deepseek-ai/FlashMLA
 */

#pragma once

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define FLASH_ASSERT(cond)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (not(cond))                                                                                                 \
        {                                                                                                              \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define FLASH_DEVICE_ASSERT(cond)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (not(cond))                                                                                                 \
        {                                                                                                              \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                                       \
            asm("trap;");                                                                                              \
        }                                                                                                              \
    } while (0)

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                             \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (COND)                                                                                                      \
        {                                                                                                              \
            constexpr static bool CONST_NAME = true;                                                                   \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            constexpr static bool CONST_NAME = false;                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                                                                   \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (NUM_SPLITS <= 32)                                                                                          \
        {                                                                                                              \
            constexpr static int NAME = 32;                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 64)                                                                                     \
        {                                                                                                              \
            constexpr static int NAME = 64;                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 96)                                                                                     \
        {                                                                                                              \
            constexpr static int NAME = 96;                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 128)                                                                                    \
        {                                                                                                              \
            constexpr static int NAME = 128;                                                                           \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 160)                                                                                    \
        {                                                                                                              \
            constexpr static int NAME = 160;                                                                           \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            FLASH_ASSERT(false);                                                                                       \
        }                                                                                                              \
    }()
