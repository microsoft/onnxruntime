// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/utils/ort_tvm_utils.h"

#include "core/codegen/common/profile.h"
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "gsl/gsl"

#include <topi/detail/extern.h>

namespace onnxruntime {
namespace tvm_codegen {

#define RETURN_DLDATATYPE_IF_MATCH(type_enum, type, type_code) \
  case type_enum:                                        \
    return {type_code, sizeof(type) * 8, 1};        \
    break;

// DLDataType: {DLDataTypeCode, bits, lanes}
DLDataType ToTvmDLDataType(MLDataType ml_type) {
  if (ml_type->IsTensorType()) {
    ml_type = ml_type->AsTensorType()->GetElementType();
  }
  auto prim_type = ml_type->AsPrimitiveDataType();
  if (prim_type == nullptr) {
    ORT_NOT_IMPLEMENTED("converting MLDataType ", ml_type, " to tvm DLDataType is not implemented");
  }

  switch (prim_type->GetDataType()) {
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_INT8, int8_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_UINT8, uint8_t, kDLUInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_INT16, int16_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_UINT16, uint16_t, kDLUInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_INT32, int32_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_UINT32, uint32_t, kDLUInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_INT64, int64_t, kDLInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_UINT64, uint64_t, kDLUInt);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_BOOL, bool, kDLUInt);

  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, float, kDLFloat);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, double, kDLFloat);
  RETURN_DLDATATYPE_IF_MATCH(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, MLFloat16, kDLFloat);
  default:
    ORT_NOT_IMPLEMENTED("converting MLDataType ", ml_type, " to tvm DLDataType is not implemented");
  }
}

tvm::Type ToTvmType(ONNX_NAMESPACE::TensorProto_DataType proto_type) {
  switch (proto_type) {
    // Note that bool is uint1 in tvm, but uint8 in ONNX, so it always require special handling
    //case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    //  return tvm::UInt(1); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return tvm::Int(16); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return tvm::Int(32); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return tvm::Int(64); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return tvm::UInt(8); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return tvm::UInt(16); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return tvm::UInt(32); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return tvm::UInt(64); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return tvm::Float(32); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return tvm::Float(64); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return tvm::Int(8); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return tvm::Float(16); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      ORT_THROW("Casting to and from strings is not supported yet."); /*break;*/
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      ORT_THROW("Cast op must have 'to' argument of type DataType"); /*break;*/
    default:
      ORT_THROW("Unexpected 'to' argument value: ", proto_type);
  }
}

tvm::Array<tvm::Expr> ShapeToTvmArray(const NodeArg* def, CodeGenContext& ctx) {
  ORT_ENFORCE(nullptr != def);
  const ONNX_NAMESPACE::TensorShapeProto* shape_proto = def->Shape();
  ORT_ENFORCE(nullptr != shape_proto);

  tvm::Array<tvm::Expr> arr;
  for (int i = 0; i < shape_proto->dim_size(); ++i) {
    arr.push_back(ShapeDimToTvmDim(shape_proto->dim(i), ctx));
  }
  return arr;
}

tvm::Expr ShapeDimToTvmDim(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim, CodeGenContext& ctx) {
  if (utils::HasDimParam(dim)) {
    return ctx.GetOrCreateDynamicDim(dim.dim_param());
  } else if (utils::HasDimValue(dim)) {
    return tvm::Expr(gsl::narrow_cast<int32_t>(dim.dim_value()));
  }
  return ctx.GetOrCreateDynamicDim(ctx.CreateUnnamedSymbol());
}

#ifdef CODEGEN_ENABLE_PROFILER
struct event_in_bracket_and_id {
  bool in_bracket;
  size_t id;
};
std::unordered_map<std::string, event_in_bracket_and_id> g_codegen_profiler_event_ids;
std::vector<std::pair<std::string, TimePoint>> g_codegen_profiler_events(1024);

TVM_REGISTER_GLOBAL("tvm.contrib.onnxruntime.profile_event")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* ret) {
      DLTensor* X = args[0];
      DLTensor* Y = args[1];
      size_t event_id = args[2];
      bool is_begin = args[3];
      if (!is_begin) {
        DCHECK(event_id < g_codegen_profiler_event_ids.size());
        profiling::Profiler::Instance().EndTimeAndRecordEvent(
            profiling::EventCategory::NODE_EVENT,
            g_codegen_profiler_events[event_id].first,
            g_codegen_profiler_events[event_id].second);
      }

      {
        CODEGEN_PROFILER_EVENT("profile_stub");
        int64_t elem_count = 1;
        for (int i = 0; i < X->ndim; ++i) {
          elem_count *= X->shape[i];
        }
        // there's overhead in this copy, so put begin after copy and end before copy
        memcpy(static_cast<char*>(Y->data) + Y->byte_offset,
               static_cast<char*>(X->data) + X->byte_offset,
               elem_count * X->dtype.bits / 8);
      }

      if (is_begin) {
        DCHECK(g_codegen_profiler_events.size() > event_id);
        DCHECK(!g_codegen_profiler_events[event_id].first.empty());
        DCHECK(g_codegen_profiler_event_ids[g_codegen_profiler_events[event_id].first].id == event_id);
        g_codegen_profiler_events[event_id].second =
            profiling::Profiler::Instance().StartTime();
      }
    });

tvm::Tensor ProfileBegin(tvm::Tensor X, const std::string& event_name) {
  size_t event_id;
  if (g_codegen_profiler_event_ids.count(event_name) == 0) {
    event_id = g_codegen_profiler_event_ids.size();
    ORT_ENFORCE(event_id < g_codegen_profiler_events.size());
  } else {
    ORT_ENFORCE(!g_codegen_profiler_event_ids[event_name].in_bracket);
    event_id = g_codegen_profiler_event_ids[event_name].id;
  }
  g_codegen_profiler_event_ids[event_name] = {true, event_id};
  g_codegen_profiler_events[event_id].first = event_name;
  return topi::detail::make_extern(
      {X->shape}, {X->dtype}, {X},
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        return topi::detail::call_packed({tvm::Expr("tvm.contrib.onnxruntime.profile_event"),
                                          topi::detail::pack_buffer(ins[0]),
                                          topi::detail::pack_buffer(outs[0]),
                                          gsl::narrow<int>(event_id),
                                          true});
      },
      event_name + "_begin", "", {})[0];
}

tvm::Tensor ProfileEnd(tvm::Tensor X, const std::string& event_name) {
  ORT_ENFORCE(g_codegen_profiler_event_ids.at(event_name).in_bracket);
  g_codegen_profiler_event_ids.at(event_name).in_bracket = false;
  size_t event_id = g_codegen_profiler_event_ids.at(event_name).id;
  ORT_ENFORCE(event_id < g_codegen_profiler_events.size());
  ORT_ENFORCE(g_codegen_profiler_events[event_id].first == event_name);
  return topi::detail::make_extern(
      {X->shape}, {X->dtype}, {X},
      [&](tvm::Array<tvm::Buffer> ins, tvm::Array<tvm::Buffer> outs) {
        return topi::detail::call_packed({tvm::Expr("tvm.contrib.onnxruntime.profile_event"),
                                          topi::detail::pack_buffer(ins[0]),
                                          topi::detail::pack_buffer(outs[0]),
                                          gsl::narrow<int>(event_id),
                                          false});
      },
      event_name + "_end", "", {})[0];
}
#endif

}  // namespace tvm_codegen
}  // namespace onnxruntime
