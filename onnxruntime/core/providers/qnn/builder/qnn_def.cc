// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_def.h"
#include <memory>
#include <cstring>
namespace onnxruntime {
namespace qnn {

size_t memscpy(void* dst, size_t dstSize, const void* src, size_t copySize) {
  if (!dst || !src || !dstSize || !copySize) return 0;

  size_t minSize = dstSize < copySize ? dstSize : copySize;

  memcpy(dst, src, minSize);

  return minSize;
}

void QnnLogStdoutCallback(const char* format,
                          QnnLog_Level_t level,
                          uint64_t timestamp,
                          va_list argument_parameter) {
  const char* levelStr = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      levelStr = " ERROR ";
      break;
    case QNN_LOG_LEVEL_WARN:
      levelStr = "WARNING";
      break;
    case QNN_LOG_LEVEL_INFO:
      levelStr = "  INFO ";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      levelStr = " DEBUG ";
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      levelStr = "VERBOSE";
      break;
    case QNN_LOG_LEVEL_MAX:
      levelStr = "UNKNOWN";
      break;
  }

  double ms = (double)timestamp / 1000000.0;
  // To avoid interleaved messages
  {
    std::lock_guard<std::mutex> lock(qnn_log_mutex_);
    fprintf(stdout, "%8.1fms [%-7s] ", ms, levelStr);
    vfprintf(stdout, format, argument_parameter);
    fprintf(stdout, "\n");
  }
}

}  // namespace qnn
}  // namespace onnxruntime
