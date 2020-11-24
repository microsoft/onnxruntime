/**
 * Copyright (c) 2016-present, Facebook, Inc.
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
// Modifications Copyright (c) Microsoft.

#pragma once

#include "onnxruntime_config.h"
// external/eigen/Eigen/src/Core/AssignEvaluator.h:86:63:
// error: enum constant in boolean context [-Werror=int-in-bool-context]

#if defined(__GNUC__)
#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if __GNUC__ >= 7
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#ifdef HAS_DEPRECATED_COPY
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif
#endif
#else
// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/Tensor.h(76):
// warning C4554: '&': check operator precedence for possible error; use parentheses to clarify precedence
// build\windows\debug\external\eigen3\unsupported\eigen\cxx11\src/Tensor/TensorStorage.h(65):
// warning C4324: structure was padded due to alignment specifier
// unsupported\eigen\cxx11\src\Tensor\TensorUInt128.h(150,0): Warning C4245: 'initializing': conversion from '__int64'
// to 'uint64_t', signed/unsigned mismatch
#pragma warning(push)
#pragma warning(disable : 4554)
#pragma warning(disable : 4324)
#pragma warning(disable : 4245)
#pragma warning(disable : 4127)
#pragma warning(disable : 6255)
#pragma warning(disable : 6294)
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif

#include "core/framework/tensor.h"
namespace onnxruntime {

class CPUMathUtil {
 public:
  /*CPUMathUtil contains some help method like generate a
        random seed. We only need a single instance for it.*/
  static CPUMathUtil& Instance() {
    static CPUMathUtil p;
    return p;
  }
  // todo: the random generate interface.
 private:
  CPUMathUtil() = default;
};


}  // namespace onnxruntime
