/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once
#include "onnx/defs/schema.h"
#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace onnxruntime {

using ONNX_NAMESPACE::OpSchema;

#define HAILO_OPERATOR_SCHEMA(name) HAILO_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define HAILO_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) HAILO_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define HAILO_OPERATOR_SCHEMA_UNIQ(Counter, name)                 \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce(  \
      op_schema_register_once##name##Counter) ONNX_UNUSED =       \
      OpSchema(#name, __FILE__, __LINE__)


void RegisterHailoSchemas() {

// Base schema for 'HailoOp'. 
// TODO: TBD - Either add inputs to this schema or add a new custom op for multiple inputs.
    HAILO_OPERATOR_SCHEMA(HailoOp)
        .SetDomain(kHailoDomain)
        .SinceVersion(1)
        .SetDoc(R"DOC(Hailo-Op)DOC")
        .Input(0, "input0", "", "T")
        .Output(0, "output0", "", "T")
        .Output(1, "output1", "", "T1", OpSchema::Optional)
        .Output(2, "output2", "", "T2", OpSchema::Optional)
        .Output(3, "output3", "", "T3", OpSchema::Optional)
        .Output(4, "output4", "", "T4", OpSchema::Optional)
        .Output(5, "output5", "", "T5", OpSchema::Optional)
        .Output(6, "output6", "", "T6", OpSchema::Optional)
        .Output(7, "output7", "", "T7", OpSchema::Optional)
        .Output(8, "output8", "", "T8", OpSchema::Optional)
        .Output(9, "output9", "", "T9", OpSchema::Optional)
        .Output(10, "output10", "", "T10", OpSchema::Optional)
        .Output(11, "output11", "", "T11", OpSchema::Optional)
        .Output(12, "output12", "", "T12", OpSchema::Optional)
        .Output(13, "output13", "", "T13", OpSchema::Optional)
        .Output(14, "output14", "", "T14", OpSchema::Optional)
        .Output(15, "output15", "", "T15", OpSchema::Optional)
        .Output(16, "output16", "", "T16", OpSchema::Optional)
        .Output(17, "output17", "", "T17", OpSchema::Optional)
        .Output(18, "output18", "", "T18", OpSchema::Optional)
        .Output(19, "output19", "", "T19", OpSchema::Optional)
        .Output(20, "output20", "", "T20", OpSchema::Optional)
        .Output(21, "output21", "", "T21", OpSchema::Optional)
        .Output(22, "output22", "", "T22", OpSchema::Optional)
        .Output(23, "output23", "", "T23", OpSchema::Optional)
        .Output(24, "output24", "", "T24", OpSchema::Optional)
        .Output(25, "output25", "", "T25", OpSchema::Optional)
        .Output(26, "output26", "", "T26", OpSchema::Optional)
        .Output(27, "output27", "", "T27", OpSchema::Optional)
        .Output(28, "output28", "", "T28", OpSchema::Optional)
        .Output(29, "output29", "", "T29", OpSchema::Optional)
        .Output(30, "output30", "", "T30", OpSchema::Optional)
        .Output(31, "output31", "", "T31", OpSchema::Optional)
        .Output(32, "output32", "", "T32", OpSchema::Optional)
        .Output(33, "output33", "", "T33", OpSchema::Optional)
        .Output(34, "output34", "", "T34", OpSchema::Optional)
        .Output(35, "output35", "", "T35", OpSchema::Optional)
        .Output(36, "output36", "", "T36", OpSchema::Optional)
        .Output(37, "output37", "", "T37", OpSchema::Optional)
        .Output(38, "output38", "", "T38", OpSchema::Optional)
        .Output(39, "output39", "", "T39", OpSchema::Optional)
        .Output(40, "output40", "", "T40", OpSchema::Optional)
        .Output(41, "output41", "", "T41", OpSchema::Optional)
        .Output(42, "output42", "", "T42", OpSchema::Optional)
        .Output(43, "output43", "", "T43", OpSchema::Optional)
        .Output(44, "output44", "", "T44", OpSchema::Optional)
        .Output(45, "output45", "", "T45", OpSchema::Optional)
        .Output(46, "output46", "", "T46", OpSchema::Optional)
        .Output(47, "output47", "", "T47", OpSchema::Optional)
        .Output(48, "output48", "", "T48", OpSchema::Optional)
        .Output(49, "output49", "", "T49", OpSchema::Optional)
        .Output(50, "output50", "", "T50", OpSchema::Optional)
        .Output(51, "output51", "", "T51", OpSchema::Optional)
        .Output(52, "output52", "", "T52", OpSchema::Optional)
        .Output(53, "output53", "", "T53", OpSchema::Optional)
        .Output(54, "output54", "", "T54", OpSchema::Optional)
        .Output(55, "output55", "", "T55", OpSchema::Optional)
        .Output(56, "output56", "", "T56", OpSchema::Optional)
        .Output(57, "output57", "", "T57", OpSchema::Optional)
        .Output(58, "output58", "", "T58", OpSchema::Optional)
        .Output(59, "output59", "", "T59", OpSchema::Optional)
        .Output(60, "output60", "", "T60", OpSchema::Optional)
        .Output(61, "output61", "", "T61", OpSchema::Optional)
        .Output(62, "output62", "", "T62", OpSchema::Optional)
        .Output(63, "output63", "", "T63", OpSchema::Optional)

        // HailoRT supported data types
        .TypeConstraint("T", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T1", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T2", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T3", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T4", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T5", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T6", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T7", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T8", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T9", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T10", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T11", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T12", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T13", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T14", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T15", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T16", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T17", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T18", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T19", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T20", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T21", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T22", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T23", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T24", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T25", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T26", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T27", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T28", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T29", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T30", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T31", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T32", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T33", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T34", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T35", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T36", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T37", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T38", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T39", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T40", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T41", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T42", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T43", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T44", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T45", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T46", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T47", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T48", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T49", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T50", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T51", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T52", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T53", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T54", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T55", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T56", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T57", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T58", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T59", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T60", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T61", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T62", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")
        .TypeConstraint("T63", {"tensor(uint8)", "tensor(uint16)", "tensor(float)"}, "")

        .Attr("hef", "", onnx::AttributeProto::STRING)
        .Attr("sorted_input_names", "", onnx::AttributeProto::STRINGS)
        .Attr("sorted_output_names", "", onnx::AttributeProto::STRINGS)
        .Attr("input_quantized", "", onnx::AttributeProto::INTS) // Represent boolean values
        .Attr("input_format_order", "", onnx::AttributeProto::INTS) // Represent the hailo_format_order_t enum
        .Attr("output_quantized", "", onnx::AttributeProto::INTS) // Represent boolean values
        .Attr("output_format_order", "", onnx::AttributeProto::INTS); // Represent the hailo_format_order_t enum
}

} // namespace onnxruntime