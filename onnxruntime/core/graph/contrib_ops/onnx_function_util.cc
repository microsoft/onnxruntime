#include "core/graph/contrib_ops/onnx_function_util.h"
#include "core/util/math.h"
#include "core/framework/float16.h"

namespace ONNX_NAMESPACE {

static uint32_t float_to_bits(float f) { return *reinterpret_cast<uint32_t*>(&f); }

static float bits_to_float(uint32_t bits) { return *reinterpret_cast<float*>(&bits); }

static uint16_t floatToHalf(float ff) {
  uint32_t floatbits = float_to_bits(ff);

  const uint32_t f32infty = {255 << 23};
  const uint32_t f16max = {(127 + 16) << 23};
  const uint32_t denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
  const uint32_t sign_mask = 0x80000000u;

  uint16_t result = static_cast<uint16_t>(0x0u);

  uint32_t sign = floatbits & sign_mask;
  floatbits ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (floatbits >= f16max) {                            // result is Inf or NaN (all exponent bits set)
    result = (floatbits > f32infty) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
  } else {                                        // (De)normalized number or zero
    if (floatbits < (113 << 23)) {                      // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      floatbits = float_to_bits(bits_to_float(floatbits) + bits_to_float(denorm_magic));

      // and one integer subtract of the bias later, we have our final float!
      result = static_cast<uint16_t>(floatbits - denorm_magic);
    } else {
      uint32_t mant_odd = (floatbits >> 13) & 1;  // resulting mantissa is odd

      // update exponent, rounding bias part 1
      floatbits += ((uint32_t)(15 - 127) << 23) + 0xfff;
      // rounding bias part 2
      floatbits += mant_odd;
      // take the bits!
      result = static_cast<uint16_t>(floatbits >> 13);
    }
  }

  result |= static_cast<uint16_t>(sign >> 16);
  return result;
}

TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
      t.add_float_data((float)value);
      break;
    case TensorProto_DataType::TensorProto_DataType_DOUBLE:
      t.add_double_data(value);
      break;
    case TensorProto_DataType::TensorProto_DataType_FLOAT16:
      t.add_int32_data(floatToHalf((float)value));
      break;
    case TensorProto_DataType::TensorProto_DataType_BFLOAT16:
      t.add_int32_data(onnxruntime::BFloat16((float)value).val);
      break;
    default:
      assert(false);
  }

  return t;
}

void BuildNodes(FunctionProto& functionProto, const std::vector<FunctionBodyHelper::NodeDef>& node_defs) {
  for (size_t i = 0; i < node_defs.size(); i++) {
    const FunctionBodyHelper::NodeDef& node = node_defs[i];
    auto* np = functionProto.add_node();

    np->set_op_type(node.op_type);
    for (const auto& inp : node.inputs) {
      np->add_input(inp);
    }
    for (const auto& o : node.outputs) {
      np->add_output(o);
    }
    for (const auto& attr : node.attributes) {
      *(np->add_attribute()) = attr.proto;
    }
  }
}

bool BuildFunctionProto(FunctionProto& functionProto, const OpSchema& schema,
                        const std::vector<FunctionBodyHelper::NodeDef>& node_defs,
                        const std::vector<OperatorSetIdProto>& relied_opsets) {
  BuildNodes(functionProto, node_defs);
  schema.BuildFunction(functionProto, relied_opsets);
  return true;
}

}  // namespace ONNX_NAMESPACE