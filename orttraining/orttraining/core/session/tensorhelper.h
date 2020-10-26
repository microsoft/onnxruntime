// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "orttraining/core/session/training_session.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <sstream>
#include <unordered_set>
#include <list>
#include <string>
#include <thread>

#include "core/common/logging/logging.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_frame.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/platform/Barrier.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/flatbuffers/ort.fbs.h"
#ifdef USE_DML  // TODO: This is necessary for the workaround in TransformGraph
#include "core/providers/dml/DmlExecutionProvider/src/GraphTransformer.h"
#endif
#include "core/session/environment.h"
#include "core/session/IOBinding.h"
#include "orttraining/models/runner/training_util.h"
#include "core/util/protobuf_parsing_utils.h"
#include "core/util/thread_utils.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/framework/customregistry.h"
#include "core/session/custom_ops.h"
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::experimental;
using namespace onnxruntime::common;
using namespace onnxruntime::training;

namespace onnxruntime {

size_t findIndex(std::vector<int64_t> dims, std::vector<int64_t> indices);

OrtValue SliceTensor(const OrtValue& orig_value, const size_t slice_id,
                    const size_t slice_axis, const size_t num_slices, TrainingSession& session_state);

}