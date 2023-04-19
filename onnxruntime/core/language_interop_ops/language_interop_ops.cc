// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "language_interop_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "pyop/pyop.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace onnxruntime {

void LoadInterOp(const std::basic_string<ORTCHAR_T>& model_uri, InterOpDomains& domains, const InterOpLogFunc& log_func) {
  int fd;

  // match the error message from model.cc to keep the nodejs tests happy.
  // as this is deprecated just cut-and-paste equivalent code for now.
  auto status = Env::Default().FileOpenRd(model_uri, fd);
  if (!status.IsOK()) {
    if (status.Category() == common::SYSTEM) {
      switch (status.Code()) {
        case ENOENT:
          status = ORT_MAKE_STATUS(ONNXRUNTIME, NO_SUCHFILE, "Load model ", ToUTF8String(model_uri),
                                   " failed. File doesn't exist");
          break;
        case EINVAL:
          status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Load model ", ToUTF8String(model_uri), " failed");
          break;
        default:
          status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "system error number ", status.Code());
      }
    }
  }

  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());

  google::protobuf::io::FileInputStream f(fd);
  f.SetCloseOnDelete(true);
  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_ENFORCE(model_proto.ParseFromZeroCopyStream(&f), "Failed to parse model proto");
  LoadInterOp(model_proto, domains, log_func);
}

void LoadInterOp(const ONNX_NAMESPACE::ModelProto& model_proto, InterOpDomains& domains, const InterOpLogFunc& log_func) {
  LoadInterOp(model_proto.graph(), domains, log_func);
}

void LoadInterOp(const ONNX_NAMESPACE::GraphProto& graph_proto, InterOpDomains& domains, const InterOpLogFunc& log_func) {
  for (int i = 0; i < graph_proto.node_size(); ++i) {
    const auto& node_proto = graph_proto.node(i);
    if (node_proto.op_type() == "PyOp") {
      auto pyop_domain = Ort::CustomOpDomain(node_proto.domain().c_str());
      pyop_domain.Add(LoadPyOp(node_proto, log_func));
      domains.push_back(std::move(pyop_domain));
    } else {
      for (int j = 0, limit = node_proto.attribute_size(); j < limit; ++j) {
        const auto& attr = node_proto.attribute(j);
        if (utils::HasGraph(attr)) {
          LoadInterOp(attr.g(), domains, log_func);  // load pyop in subgraph
        }
      }  // for
    }    // else
  }      // for
}
}  // namespace onnxruntime
