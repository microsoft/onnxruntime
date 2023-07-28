// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensorprotoutils.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/path.h"
#include "core/common/path_string.h"
#include "core/common/status.h"
#include "core/framework/framework_common.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"

namespace onnxruntime {
namespace training {

#if !defined(ORT_MINIMAL_BUILD)

/**
 * @brief Open file descriptor and call use_fn
 *
 * @tparam TUseFileFn
 * @param path file path
 * @param readonly open mode.
 * @param use_fn function taking file descriptor as inputs.
 * @return common::Status
 */
template <typename TUseFileFn>
common::Status WithOpenFile(const PathString& path, bool readonly, TUseFileFn use_fn) {
  int fd;
  if (readonly) {
    ORT_RETURN_IF_ERROR(Env::Default().FileOpenRd(path, fd));
  } else {
    ORT_RETURN_IF_ERROR(Env::Default().FileOpenWr(path, fd));
  }

  Status use_fn_status{};
  try {
    use_fn_status = use_fn(fd);
  } catch (std::exception& e) {
    use_fn_status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, e.what());
  }

  Status close_status = Env::Default().FileClose(fd);
  return !use_fn_status.IsOK() ? use_fn_status : close_status;
}

#endif

/**
 * @brief Create OrtValues From TensorProto objects
 *
 * @param tensor_protos vector of TensorProto
 * @param name_to_ort_value saved results.
 * @return Status
 */
Status CreateOrtValuesFromTensorProtos(
    const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    NameMLValMap& name_to_ort_value);

}  // namespace training
}  // namespace onnxruntime
