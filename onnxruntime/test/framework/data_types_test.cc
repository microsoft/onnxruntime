// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <typeinfo>
#include <cmath>

#include "core/common/inlined_containers.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/float16.h"
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

#include "core/util/math.h"
#include <ostream>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/data_type_utils.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {

template <typename K, typename V>
struct TestMap {
  using key_type = K;
  using mapped_type = V;
};

// Try recursive type registration and compatibility tests
using VectorInt64 = std::vector<int64_t>;
#if !defined(DISABLE_ML_OPS)
using TestMapToMapInt64ToFloat = TestMap<int64_t, MapInt64ToFloat>;
using TestMapStringToVectorInt64 = TestMap<std::string, VectorInt64>;
#endif

// Trial to see if we resolve the setter properly
// a map with a key that has not been registered in data_types.cc
using TestMapMLFloat16ToFloat = TestMap<MLFloat16, float>;

template <typename T>
struct TestSequence {
  using value_type = T;
};

using VectorString = std::vector<std::string>;
using TestSequenceOfSequence = TestSequence<VectorString>;

/// Adding an Opaque type with type parameters
struct TestOpaqueType_1 {};
struct TestOpaqueType_2 {};

// String arrays must be extern to make them unique
// so the instantiated template would produce a unique type as well.
extern const char TestOpaqueDomain_1[] = "test_domain_1";
extern const char TestOpaqueName_1[] = "test_name_1";

extern const char TestOpaqueDomain_2[] = "test_domain_2";
extern const char TestOpaqueName_2[] = "test_name_2";

extern const char TestOpaqueEmpty[] = "";

struct TestOpaqueDomainOnly {};
struct TestOpaqueNameOnly {};
struct TestOpaqueNoNames {};

// Register Maps using Opaque types as values. Note that we
// use the same cpp runtime types but due to Opaque type domain, name
// and optional parameters we produce separate MLDataTypes that are NOT
// compatible with each other.
#if !defined(DISABLE_ML_OPS)
using MyOpaqueMapCpp_1 = std::map<int64_t, TestOpaqueType_1>;
using MyOpaqueMapCpp_2 = std::map<int64_t, TestOpaqueType_2>;
#endif

// Register Sequence as containing an Opaque type
using MyOpaqueSeqCpp_1 = std::vector<TestOpaqueType_1>;
using MyOpaqueSeqCpp_2 = std::vector<TestOpaqueType_2>;

#if !defined(DISABLE_ML_OPS)
ORT_REGISTER_MAP(MyOpaqueMapCpp_1);
ORT_REGISTER_MAP(MyOpaqueMapCpp_2);

ORT_REGISTER_MAP(TestMapToMapInt64ToFloat);
ORT_REGISTER_MAP(TestMapStringToVectorInt64);
ORT_REGISTER_MAP(TestMapMLFloat16ToFloat);
#endif

ORT_REGISTER_SEQ(MyOpaqueSeqCpp_1);
ORT_REGISTER_SEQ(MyOpaqueSeqCpp_2);
ORT_REGISTER_SEQ(TestSequenceOfSequence);
ORT_REGISTER_SEQ(VectorString);
ORT_REGISTER_SEQ(VectorInt64);

ORT_REGISTER_OPAQUE_TYPE(TestOpaqueType_1, TestOpaqueDomain_1, TestOpaqueName_1);
ORT_REGISTER_OPAQUE_TYPE(TestOpaqueType_2, TestOpaqueDomain_2, TestOpaqueName_2);
// Special cases
ORT_REGISTER_OPAQUE_TYPE(TestOpaqueDomainOnly, TestOpaqueDomain_1, TestOpaqueEmpty);
ORT_REGISTER_OPAQUE_TYPE(TestOpaqueNameOnly, TestOpaqueEmpty, TestOpaqueName_1);
ORT_REGISTER_OPAQUE_TYPE(TestOpaqueNoNames, TestOpaqueEmpty, TestOpaqueEmpty);

#define REGISTER_ONNX_PROTO(TYPE)                      \
  {                                                    \
    MLDataType mltype = DataTypeImpl::GetType<TYPE>(); \
    DataTypeImpl::RegisterDataType(mltype);            \
  }

void RegisterTestTypes() {
#if !defined(DISABLE_ML_OPS)
  REGISTER_ONNX_PROTO(MyOpaqueMapCpp_1);
  REGISTER_ONNX_PROTO(MyOpaqueMapCpp_2);

  REGISTER_ONNX_PROTO(TestMapToMapInt64ToFloat);
  REGISTER_ONNX_PROTO(TestMapStringToVectorInt64);
  REGISTER_ONNX_PROTO(TestMapMLFloat16ToFloat);
#endif

  REGISTER_ONNX_PROTO(MyOpaqueSeqCpp_1);
  REGISTER_ONNX_PROTO(MyOpaqueSeqCpp_2);
  REGISTER_ONNX_PROTO(TestSequenceOfSequence);

  REGISTER_ONNX_PROTO(TestOpaqueType_1);
  REGISTER_ONNX_PROTO(TestOpaqueType_2);
  REGISTER_ONNX_PROTO(TestOpaqueDomainOnly);
  REGISTER_ONNX_PROTO(TestOpaqueNameOnly);
  REGISTER_ONNX_PROTO(TestOpaqueNoNames);
}

namespace test {
using namespace ONNX_NAMESPACE;

template <int... dims>
struct DimSetter;

template <>
struct DimSetter<> {
  static void set(TensorShapeProto&) {}
};

inline void AddDim(TensorShapeProto& proto, int d) {
  proto.add_dim()->set_dim_value(d);
}

template <int d>
struct DimSetter<d> {
  static void set(TensorShapeProto& proto) {
    AddDim(proto, d);
  }
};

template <int d, int... dims>
struct DimSetter<d, dims...> {
  static void set(TensorShapeProto& proto) {
    AddDim(proto, d);
    DimSetter<dims...>::set(proto);
  }
};

template <int... dims>
struct TensorShapeTypeProto {
  TensorShapeTypeProto() {
    DimSetter<dims...>::set(proto);
  }
  TensorShapeProto proto;
};

template <>
struct TensorShapeTypeProto<> {
  TensorShapeProto proto;
};

template <TensorProto_DataType T>
struct TensorTypeProto {
  TensorTypeProto() {
    proto.mutable_tensor_type()->set_elem_type(T);
  }
  TypeProto proto;
};

template <TensorProto_DataType T>
struct SparseTensorTypeProto {
  SparseTensorTypeProto() {
    proto.mutable_sparse_tensor_type()->set_elem_type(T);
  }
  void SetShape(const TensorShapeProto& shape) {
    proto.mutable_sparse_tensor_type()->mutable_shape()->CopyFrom(shape);
  }
  void SetShape(TensorShapeProto&& shape) {
    *proto.mutable_sparse_tensor_type()->mutable_shape() = std::move(shape);
  }
  void ClearShape() {
    proto.mutable_sparse_tensor_type()->clear_shape();
  }
  TypeProto proto;
};

template <TensorProto_DataType key, TensorProto_DataType value>
struct MapTypeProto {
  MapTypeProto() {
    proto.mutable_map_type()->set_key_type(key);
    proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(value);
  }
  TypeProto proto;
};

class DataTypeTest : public testing::Test {
 public:
  static void SetUpTestCase() {
    // xcTest run test case by case, so the SetUp needs to be reentrant.
    static std::atomic<bool> loaded(false);
    if (!loaded.load()) {
      loaded.store(true);
      RegisterTestTypes();
    }
  }
};

TEST_F(DataTypeTest, OpaqueRegistrationTest) {
  // No parameters
  TypeProto opaque_proto_1;
  auto* mop = opaque_proto_1.mutable_opaque_type();
  mop->mutable_domain()->assign(TestOpaqueDomain_1);
  mop->mutable_name()->assign(TestOpaqueName_1);

  EXPECT_TRUE(DataTypeImpl::GetType<TestOpaqueType_1>()->IsCompatible(opaque_proto_1));
  // OpaqueType_2 has the same domain and name but also has parameters
  // so it is not compatible
  EXPECT_FALSE(DataTypeImpl::GetType<TestOpaqueType_2>()->IsCompatible(opaque_proto_1));

  // Now change domain and name for that of OpaqueType_3
  // now we are supposed to be compatible with OpaqueType_2 but not
  // OpaqueType_1
  mop->mutable_domain()->assign(TestOpaqueDomain_2);
  mop->mutable_name()->assign(TestOpaqueName_2);
  EXPECT_FALSE(DataTypeImpl::GetType<TestOpaqueType_1>()->IsCompatible(opaque_proto_1));
  EXPECT_TRUE(DataTypeImpl::GetType<TestOpaqueType_2>()->IsCompatible(opaque_proto_1));

  // assign back original domain/name and add params
  mop->mutable_domain()->assign(TestOpaqueDomain_2);
  mop->mutable_name()->assign(TestOpaqueName_2);

  EXPECT_FALSE(DataTypeImpl::GetType<TestOpaqueType_1>()->IsCompatible(opaque_proto_1));
  EXPECT_TRUE(DataTypeImpl::GetType<TestOpaqueType_2>()->IsCompatible(opaque_proto_1));

  auto op_ml1 = DataTypeImpl::GetType<TestOpaqueType_1>();
  auto op_ml2 = DataTypeImpl::GetType<TestOpaqueType_2>();
  // Test IsOpaqueType
  EXPECT_TRUE(utils::IsOpaqueType(op_ml1, TestOpaqueDomain_1, TestOpaqueName_1));
  EXPECT_FALSE(utils::IsOpaqueType(op_ml1, TestOpaqueDomain_1, TestOpaqueName_2));
  EXPECT_TRUE(utils::IsOpaqueType(op_ml2, TestOpaqueDomain_2, TestOpaqueName_2));
  EXPECT_FALSE(utils::IsOpaqueType(op_ml2, TestOpaqueDomain_1, TestOpaqueName_2));
  EXPECT_FALSE(utils::IsOpaqueType(DataTypeImpl::GetTensorType<float>(), TestOpaqueDomain_1, TestOpaqueName_1));

#if !defined(DISABLE_ML_OPS)
  utils::ContainerChecker c_checker(DataTypeImpl::GetType<MyOpaqueMapCpp_1>());
  EXPECT_TRUE(c_checker.IsMap());
  bool result = c_checker.IsMapOf<int64_t, TestOpaqueType_1>();
  EXPECT_TRUE(result);
#endif
}

#if !defined(DISABLE_ML_OPS)
TEST_F(DataTypeTest, MapStringStringTest) {
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  auto ml_str_str = DataTypeImpl::GetType<MapStringToString>();
  EXPECT_TRUE(DataTypeImpl::GetTensorType<float>()->IsCompatible(tensor_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetTensorType<uint64_t>()->IsCompatible(tensor_type.proto));
  EXPECT_FALSE(ml_str_str->IsCompatible(tensor_type.proto));
  utils::ContainerChecker c_checker(ml_str_str);
  bool result = c_checker.IsMapOf<std::string, std::string>();
  EXPECT_TRUE(result);
  result = c_checker.IsMapOf<std::string, int64_t>();
  EXPECT_FALSE(result);

  utils::ContainerChecker c_checker1(DataTypeImpl::GetTensorType<float>());
  result = c_checker1.IsMapOf<std::string, int64_t>();
  EXPECT_FALSE(result);

  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_STRING> maps2s_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  EXPECT_TRUE(ml_str_str->IsCompatible(maps2s_type.proto));
  EXPECT_FALSE(ml_str_str->IsCompatible(maps2i_type.proto));
}

TEST_F(DataTypeTest, MapStringInt64Test) {
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_STRING> maps2s_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(maps2s_type.proto));
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(maps2i_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToInt64>()->IsCompatible(tensor_type.proto));

  utils::ContainerChecker c_checker(DataTypeImpl::GetType<MapStringToInt64>());
  bool result = c_checker.IsMapOf<std::string, int64_t>();
  EXPECT_TRUE(result);
}

TEST_F(DataTypeTest, MapStringFloatTest) {
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_FLOAT> maps2f_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(maps2f_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(maps2i_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToFloat>()->IsCompatible(tensor_type.proto));
}

TEST_F(DataTypeTest, MapStringDoubleTest) {
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_DOUBLE> maps2d_type;
  MapTypeProto<TensorProto_DataType_STRING, TensorProto_DataType_INT64> maps2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(maps2d_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(maps2i_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapStringToDouble>()->IsCompatible(tensor_type.proto));
}

TEST_F(DataTypeTest, MapInt64StringTest) {
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_STRING> mapi2s_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2s_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2i_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(tensor_type.proto));
}

TEST_F(DataTypeTest, MapInt64DoubleTest) {
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;
  EXPECT_TRUE(DataTypeImpl::GetType<MapInt64ToDouble>()->IsCompatible(mapi2d_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(mapi2i_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<MapInt64ToString>()->IsCompatible(tensor_type.proto));
}

TEST_F(DataTypeTest, RecursiveMapTest) {
  TypeProto map_int64_to_map_int64_to_float;
  auto* mut_map = map_int64_to_map_int64_to_float.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map = mut_map->mutable_value_type()->mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto map_string_to_vector_of_int64;
  mut_map = map_string_to_vector_of_int64.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_STRING);
  mut_map->mutable_value_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  EXPECT_TRUE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_int64_to_map_int64_to_float));
  EXPECT_FALSE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_string_to_vector_of_int64));

  EXPECT_TRUE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_int64_to_map_int64_to_float));
  EXPECT_FALSE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(map_string_to_vector_of_int64));

  // Map that contains an Opaque_1
  const auto* op1_proto = DataTypeImpl::GetType<TestOpaqueType_1>();
  TypeProto unod_map_int64_to_op1;
  mut_map = unod_map_int64_to_op1.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map->mutable_value_type()->CopyFrom(*op1_proto->GetTypeProto());
  EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueMapCpp_1>()->IsCompatible(unod_map_int64_to_op1));

  // Map that contains an Opaque_2
  const auto* op2_proto = DataTypeImpl::GetType<TestOpaqueType_2>();
  TypeProto unod_map_int64_to_op2;
  mut_map = unod_map_int64_to_op2.mutable_map_type();
  mut_map->set_key_type(TensorProto_DataType_INT64);
  mut_map->mutable_value_type()->CopyFrom(*op2_proto->GetTypeProto());
  EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueMapCpp_2>()->IsCompatible(unod_map_int64_to_op2));
}
#endif  // !defined(DISABLE_ML_OPS)

TEST_F(DataTypeTest, RecursiveVectorTest) {
  TypeProto seq_of_seq_string;
  auto* mut_seq = seq_of_seq_string.mutable_sequence_type();
  mut_seq = mut_seq->mutable_elem_type()->mutable_sequence_type();
  mut_seq->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_STRING);

  EXPECT_TRUE(DataTypeImpl::GetType<TestSequenceOfSequence>()->IsCompatible(seq_of_seq_string));
#if !defined(DISABLE_ML_OPS)
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(seq_of_seq_string));
#endif
}

#if !defined(DISABLE_ML_OPS)
TEST_F(DataTypeTest, VectorMapStringToFloatTest) {
  TypeProto vector_map_string_to_float;
  vector_map_string_to_float.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->set_key_type(TensorProto_DataType_STRING);
  vector_map_string_to_float.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;

  EXPECT_TRUE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(vector_map_string_to_float));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(mapi2d_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(mapi2i_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(tensor_type.proto));
  utils::ContainerChecker c_check(DataTypeImpl::GetType<VectorMapStringToFloat>());
  bool result = c_check.IsSequenceOf<MapStringToFloat>();
  EXPECT_TRUE(result);
}

TEST_F(DataTypeTest, VectorMapInt64ToFloatTest) {
  TypeProto type_proto;
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->set_key_type(TensorProto_DataType_INT64);
  type_proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_DOUBLE> mapi2d_type;
  MapTypeProto<TensorProto_DataType_INT64, TensorProto_DataType_INT64> mapi2i_type;
  TensorTypeProto<TensorProto_DataType_FLOAT> tensor_type;

  EXPECT_TRUE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(type_proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(mapi2d_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(mapi2i_type.proto));
  EXPECT_FALSE(DataTypeImpl::GetType<VectorMapInt64ToFloat>()->IsCompatible(tensor_type.proto));
}
#endif  // !defined(DISABLE_ML_OPS)

TEST_F(DataTypeTest, MlFloat16ConvertFloatToMLFloat16) {
  // Test data type
  {
    constexpr float sample = 1.0f;
    const MLFloat16 flt16(sample);
    auto int_rep = flt16.val;
    const auto flt_from_int = MLFloat16::FromBits(int_rep);
    const double diff = std::fabs(sample - flt_from_int.ToFloat());
    if (diff > FLT_EPSILON || (std::isnan(diff) && !std::isnan(sample))) {
      EXPECT_TRUE(false);
    }
  }
  // Test bulk conversion
  {
    float sample[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<MLFloat16> converted;
    std::transform(std::begin(sample), std::end(sample), std::back_inserter(converted),
                   [](float fl) { return MLFloat16(fl); });
    for (size_t i = 0; i < sizeof(sample) / sizeof(float); ++i) {
      const double diff = std::fabs(sample[i] - converted[i].ToFloat());
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        FAIL();
      }
    }

    std::vector<float> back_converted;
    std::transform(converted.cbegin(), converted.cend(), std::back_inserter(back_converted),
                   [](const MLFloat16 ml) { return (float)ml; });
    for (size_t i = 0; i < sizeof(sample) / sizeof(float); ++i) {
      const double diff = std::fabs(sample[i] - back_converted[i]);
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        FAIL();
      }
    }
  }
}

TEST_F(DataTypeTest, MLFloat16Zeros) {
  const auto positive_zero = MLFloat16::FromBits(0U);
  EXPECT_FALSE(positive_zero.IsNegative());
  const float float_positive_zero = static_cast<float>(positive_zero);
  EXPECT_EQ(+0.0f, float_positive_zero);
  EXPECT_FALSE(std::signbit(float_positive_zero));

  const auto negative_zero = positive_zero.Negate();
  EXPECT_TRUE(negative_zero.IsNegative());
  const float float_positive_negzero = static_cast<float>(negative_zero);
  EXPECT_EQ(-0.0f, float_positive_negzero);
  EXPECT_TRUE(std::signbit(float_positive_negzero));

  EXPECT_TRUE(positive_zero.IsNaNOrZero());
  EXPECT_TRUE(negative_zero.IsNaNOrZero());
}

TEST_F(DataTypeTest, MLFloat16Comparision) {
  const MLFloat16 left = MLFloat16(-33.33f);
  const MLFloat16 left_same = MLFloat16(-33.33f);
  const MLFloat16 right = MLFloat16(66.66f);
  const MLFloat16 right_same = MLFloat16(66.66f);

  EXPECT_TRUE(MLFloat16::Epsilon < right);

  EXPECT_EQ(left, left_same);
  EXPECT_NE(left, left_same.Negate());

  EXPECT_EQ(right, right_same);
  EXPECT_NE(right, right_same.Negate());

  EXPECT_LT(left, right);
  EXPECT_LT(right.Negate(), left);
  EXPECT_LT(left.Negate(), right);
}

TEST_F(DataTypeTest, MLFloat16TestNAN) {
  const MLFloat16 fp16NANFromSingle(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(fp16NANFromSingle.IsNaN());
  EXPECT_TRUE(fp16NANFromSingle.IsNaNOrZero());

  // NaN are not equal to each other
  EXPECT_NE(MLFloat16::NaN, fp16NANFromSingle);

  const float NanFromBFloat16 = fp16NANFromSingle.ToFloat();
  EXPECT_TRUE(std::isnan(NanFromBFloat16));

  EXPECT_FALSE(MLFloat16::FromBits(MLFloat16::kMaxValueBits).IsNaN());
}

TEST_F(DataTypeTest, MLFloat16NaNComparision) {
  EXPECT_FALSE(MLFloat16::NaN < MLFloat16::NaN);
  EXPECT_FALSE(MLFloat16::NaN == MLFloat16::NaN);

  EXPECT_FALSE(MLFloat16::MaxValue < MLFloat16::NaN);
  EXPECT_FALSE(MLFloat16::MaxValue == MLFloat16::NaN);
  EXPECT_FALSE(MLFloat16::MinValue < MLFloat16::NaN);
  EXPECT_FALSE(MLFloat16::NaN < MLFloat16::MaxValue);

  EXPECT_TRUE(MLFloat16::MinValue < MLFloat16::MaxValue);
}

TEST_F(DataTypeTest, MLFloat16Infinity) {
  EXPECT_FALSE(MLFloat16::MinValue.IsInfinity());
  EXPECT_FALSE(MLFloat16::MaxValue.IsInfinity());
  EXPECT_TRUE(MLFloat16::MaxValue.IsFinite());

  const MLFloat16 pos_infinity_from_float(std::numeric_limits<float>::infinity());
  EXPECT_TRUE(pos_infinity_from_float.IsInfinity());
  EXPECT_FALSE(pos_infinity_from_float.IsFinite());
  EXPECT_FALSE(pos_infinity_from_float.IsNegative());

  const MLFloat16 neg_infinity_from_float(-std::numeric_limits<float>::infinity());
  EXPECT_TRUE(neg_infinity_from_float.IsInfinity());
  EXPECT_FALSE(neg_infinity_from_float.IsFinite());
  EXPECT_TRUE(neg_infinity_from_float.IsNegative());

  const float pos_infinity_from_bfloat16 = static_cast<float>(MLFloat16::Infinity);
  EXPECT_TRUE(std::isinf(pos_infinity_from_bfloat16));
  EXPECT_TRUE(!std::signbit(pos_infinity_from_bfloat16));
}

TEST_F(DataTypeTest, MLFloat16NormalSubnormal) {
  EXPECT_FALSE(MLFloat16::Infinity.IsNormal());
  EXPECT_TRUE(MLFloat16(45.6f).IsNormal());
  EXPECT_FALSE(MLFloat16(45.6f).IsSubnormal());

  // 0b0_0000_0000_000_0001 ~0.000000059604645
  constexpr uint16_t min_subnormal_bits = 0x0001;
  const MLFloat16 smallest_subnormal = MLFloat16::FromBits(min_subnormal_bits);
  EXPECT_TRUE(smallest_subnormal.IsSubnormal());
  EXPECT_FALSE(smallest_subnormal.IsNormal());

  // float smallest positive subnormal is ~1.40129846432481707092E-45, and
  // in float the same number above would be normal
  const float float_from_smallest_subnormal = static_cast<float>(smallest_subnormal);
  EXPECT_TRUE(std::isnormal(float_from_smallest_subnormal));

  // 0b0_0000_0000_111_1111; ~0.000060975552
  constexpr uint16_t max_subnormal_bits = 0x007F;
  const MLFloat16 largest_subnormal = MLFloat16::FromBits(max_subnormal_bits);
  EXPECT_TRUE(largest_subnormal.IsSubnormal());
  EXPECT_FALSE(largest_subnormal.IsNormal());

  // However, in float the same number above would be normal
  const float float_from_largest_subnormal = static_cast<float>(largest_subnormal);
  EXPECT_TRUE(std::isnormal(float_from_largest_subnormal));
}

TEST_F(DataTypeTest, BFloat16ConvertFloatToBFloat16) {
  // Test data type
  {
    constexpr float sample = 1.0f;
    const BFloat16 flt16(sample);
    auto int_rep = flt16.val;
    const auto flt_from_int = BFloat16::FromBits(int_rep);
    const double diff = std::fabs(sample - flt_from_int.ToFloat());
    if (diff > FLT_EPSILON || (std::isnan(diff) && !std::isnan(sample))) {
      EXPECT_TRUE(false);
    }
  }
  // Test bulk conversion
  {
    float sample[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    BFloat16 converted[sizeof(sample) / sizeof(float)];
    static_assert(sizeof(sample) / sizeof(float) == sizeof(converted) / sizeof(BFloat16), "Must have the same count");
    FloatToBFloat16(sample, converted, sizeof(sample) / sizeof(float));
    for (size_t i = 0; i < sizeof(sample) / sizeof(float); ++i) {
      const double diff = std::fabs(sample[i] - converted[i].ToFloat());
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        EXPECT_TRUE(false);
      }
    }

    float back_converted[sizeof(sample) / sizeof(float)];
    BFloat16ToFloat(converted, back_converted, sizeof(sample) / sizeof(float));
    for (size_t i = 0; i < sizeof(sample) / sizeof(float); ++i) {
      const double diff = std::fabs(sample[i] - back_converted[i]);
      if ((std::isnan(diff) && !std::isnan(sample[i])) || diff > FLT_EPSILON) {
        EXPECT_TRUE(false);
      }
    }
  }
}

TEST_F(DataTypeTest, BFloat16Zeros) {
  const auto positive_zero = BFloat16::FromBits(0U);
  EXPECT_FALSE(positive_zero.IsNegative());
  const float float_positive_zero = static_cast<float>(positive_zero);
  EXPECT_EQ(+0.0f, float_positive_zero);
  EXPECT_FALSE(std::signbit(float_positive_zero));

  const auto negative_zero = positive_zero.Negate();
  EXPECT_TRUE(negative_zero.IsNegative());
  const float float_positive_negzero = static_cast<float>(negative_zero);
  EXPECT_EQ(-0.0f, float_positive_negzero);
  EXPECT_TRUE(std::signbit(float_positive_negzero));

  EXPECT_TRUE(positive_zero.IsNaNOrZero());
  EXPECT_TRUE(negative_zero.IsNaNOrZero());
}

TEST_F(DataTypeTest, BFloat16Comparision) {
  const BFloat16 left = BFloat16(-33.33f);
  const BFloat16 left_same = BFloat16(-33.33f);
  const BFloat16 right = BFloat16(66.66f);
  const BFloat16 right_same = BFloat16(66.66f);

  EXPECT_TRUE(BFloat16::Epsilon < right);

  EXPECT_EQ(left, left_same);
  EXPECT_NE(left, left_same.Negate());

  EXPECT_EQ(right, right_same);
  EXPECT_NE(right, right_same.Negate());

  EXPECT_LT(left, right);
  EXPECT_LT(right.Negate(), left);
  EXPECT_LT(left.Negate(), right);
}

TEST_F(DataTypeTest, BFloat16TestNAN) {
  const BFloat16 fp16NANFromSingle = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(fp16NANFromSingle.IsNaN());
  EXPECT_TRUE(fp16NANFromSingle.IsNaNOrZero());
  // NaN are not equal to each other
  EXPECT_NE(BFloat16::NaN, fp16NANFromSingle);

  float NanFromBFloat16 = fp16NANFromSingle.ToFloat();
  EXPECT_TRUE(std::isnan(NanFromBFloat16));

  EXPECT_FALSE(BFloat16::FromBits(BFloat16::kMaxValueBits).IsNaN());
}

TEST_F(DataTypeTest, BFloat16NaNComparision) {
  EXPECT_FALSE(BFloat16::NaN < BFloat16::NaN);
  EXPECT_FALSE(BFloat16::NaN == BFloat16::NaN);

  EXPECT_FALSE(BFloat16::MaxValue < BFloat16::NaN);
  EXPECT_FALSE(BFloat16::MaxValue == BFloat16::NaN);
  EXPECT_FALSE(BFloat16::MinValue < BFloat16::NaN);
  EXPECT_FALSE(BFloat16::NaN < BFloat16::MaxValue);

  EXPECT_TRUE(BFloat16::MinValue < BFloat16::MaxValue);
}

TEST_F(DataTypeTest, BFloat16Infinity) {
  EXPECT_FALSE(BFloat16::MinValue.IsInfinity());
  EXPECT_FALSE(BFloat16::MaxValue.IsInfinity());
  EXPECT_TRUE(BFloat16::MaxValue.IsFinite());

  const BFloat16 pos_infinity_from_float = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(pos_infinity_from_float.IsInfinity());
  EXPECT_FALSE(pos_infinity_from_float.IsFinite());
  EXPECT_FALSE(pos_infinity_from_float.IsNegative());

  const BFloat16 neg_infinity_from_float = -std::numeric_limits<float>::infinity();
  EXPECT_TRUE(neg_infinity_from_float.IsInfinity());
  EXPECT_FALSE(neg_infinity_from_float.IsFinite());
  EXPECT_TRUE(neg_infinity_from_float.IsNegative());
  EXPECT_TRUE(std::signbit(neg_infinity_from_float.ToFloat()));

  const float pos_infinity_from_bfloat16 = static_cast<float>(BFloat16::Infinity);
  EXPECT_TRUE(std::isinf(pos_infinity_from_bfloat16));
  EXPECT_TRUE(!std::signbit(pos_infinity_from_bfloat16));
}

TEST_F(DataTypeTest, BFloat16NormalSubnormal) {
  EXPECT_FALSE(BFloat16::Infinity.IsNormal());
  EXPECT_TRUE(BFloat16(45.6f).IsNormal());
  EXPECT_FALSE(BFloat16(45.6f).IsSubnormal());

  // 0b0_0000_0000_000_0001
  constexpr uint16_t min_subnormal_bits = 0x0001;
  const BFloat16 smallest_subnormal = BFloat16::FromBits(min_subnormal_bits);
  EXPECT_TRUE(smallest_subnormal.IsSubnormal());
  EXPECT_FALSE(smallest_subnormal.IsNormal());

  const float float_from_smallest_subnormal = (float)smallest_subnormal;
  EXPECT_FALSE(std::isnormal(float_from_smallest_subnormal));

  // 0b0_0000_0000_111_1111;
  constexpr uint16_t max_subnormal_bits = 0x007F;
  const BFloat16 largest_subnormal = BFloat16::FromBits(max_subnormal_bits);
  EXPECT_TRUE(largest_subnormal.IsSubnormal());
  EXPECT_FALSE(largest_subnormal.IsNormal());

  const float float_from_largest_subnormal = (float)largest_subnormal;
  EXPECT_FALSE(std::isnormal(float_from_largest_subnormal));
}

TEST_F(DataTypeTest, DataUtilsTest) {
  using namespace ONNX_NAMESPACE::Utils;
  // Test simple seq
  {
    const std::string seq_float("seq(tensor(float))");
    const auto* seq_proto = DataTypeImpl::GetSequenceTensorType<float>()->GetTypeProto();
    EXPECT_NE(seq_proto, nullptr);
    DataType seq_dt = DataTypeUtils::ToType(*seq_proto);
    EXPECT_NE(seq_dt, nullptr);
    EXPECT_EQ(seq_float, *seq_dt);
    DataType seq_from_str = DataTypeUtils::ToType(*seq_dt);
    // Expect internalized strings
    EXPECT_EQ(seq_dt, seq_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(seq_dt);
    EXPECT_TRUE(DataTypeImpl::GetSequenceTensorType<float>()->IsCompatible(from_dt_proto));
  }
  // Test Tensor
  {
    const std::string tensor_uint64("tensor(uint64)");
    const auto* ten_proto = DataTypeImpl::GetTensorType<uint64_t>()->GetTypeProto();
    EXPECT_NE(ten_proto, nullptr);
    DataType ten_dt = DataTypeUtils::ToType(*ten_proto);
    EXPECT_NE(ten_dt, nullptr);
    EXPECT_EQ(tensor_uint64, *ten_dt);
    DataType ten_from_str = DataTypeUtils::ToType(*ten_dt);
    // Expect internalized strings
    EXPECT_EQ(ten_dt, ten_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(ten_dt);
    EXPECT_TRUE(DataTypeImpl::GetTensorType<uint64_t>()->IsCompatible(from_dt_proto));
  }
  // Test Tensor with bfloat16
  {
    const std::string tensor_uint64("tensor(bfloat16)");
    const auto* ten_proto = DataTypeImpl::GetTensorType<BFloat16>()->GetTypeProto();
    EXPECT_NE(ten_proto, nullptr);
    DataType ten_dt = DataTypeUtils::ToType(*ten_proto);
    EXPECT_NE(ten_dt, nullptr);
    EXPECT_EQ(tensor_uint64, *ten_dt);
    DataType ten_from_str = DataTypeUtils::ToType(*ten_dt);
    // Expect internalized strings
    EXPECT_EQ(ten_dt, ten_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(ten_dt);
    EXPECT_TRUE(DataTypeImpl::GetTensorType<BFloat16>()->IsCompatible(from_dt_proto));
  }
  // SparseTensor
  // Currently test only with proto, no MLDataType yet.
  {
    const std::string tensor_uint64("sparse_tensor(uint64)");
    // We expect that the above string will be matched in both cases
    // where we have shape and where we don't
    SparseTensorTypeProto<TensorProto_DataType_UINT64> sparse_proto;
    DataType ten_dt = DataTypeUtils::ToType(sparse_proto.proto);
    EXPECT_NE(ten_dt, nullptr);
    EXPECT_EQ(tensor_uint64, *ten_dt);
    DataType ten_from_str = DataTypeUtils::ToType(*ten_dt);
    // Expect internalized strings
    EXPECT_EQ(ten_dt, ten_from_str);

    // Now add empty shape, we expect the same string
    TensorShapeTypeProto<> shape_no_dims;
    sparse_proto.SetShape(shape_no_dims.proto);
    ten_dt = DataTypeUtils::ToType(sparse_proto.proto);
    EXPECT_NE(ten_dt, nullptr);
    EXPECT_EQ(tensor_uint64, *ten_dt);
    ten_from_str = DataTypeUtils::ToType(*ten_dt);
    // Expect internalized strings
    EXPECT_EQ(ten_dt, ten_from_str);

    // Now add shape with dimensions, we expect no difference
    sparse_proto.ClearShape();
    TensorShapeTypeProto<10, 12> shape_with_dim;
    sparse_proto.SetShape(shape_with_dim.proto);
    ten_dt = DataTypeUtils::ToType(sparse_proto.proto);
    EXPECT_NE(ten_dt, nullptr);
    EXPECT_EQ(tensor_uint64, *ten_dt);
    ten_from_str = DataTypeUtils::ToType(*ten_dt);
    // Expect internalized strings
    EXPECT_EQ(ten_dt, ten_from_str);
  }

#if !defined(DISABLE_ML_OPS)
  // Test Simple map
  {
    const std::string map_string_string("map(string,tensor(string))");
    const auto* map_proto = DataTypeImpl::GetType<MapStringToString>()->GetTypeProto();
    EXPECT_NE(map_proto, nullptr);
    DataType map_dt = DataTypeUtils::ToType(*map_proto);
    EXPECT_NE(map_dt, nullptr);
    EXPECT_EQ(map_string_string, *map_dt);
    DataType map_from_str = DataTypeUtils::ToType(*map_dt);
    // Expect internalized strings
    EXPECT_EQ(map_dt, map_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(map_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<MapStringToString>()->IsCompatible(from_dt_proto));
  }

  // Test map with recursive value
  {
    const std::string map_int_map_int_float("map(int64,map(int64,tensor(float)))");
    const auto* map_proto = DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->GetTypeProto();
    EXPECT_NE(map_proto, nullptr);
    DataType map_dt = DataTypeUtils::ToType(*map_proto);
    EXPECT_NE(map_dt, nullptr);
    EXPECT_EQ(map_int_map_int_float, *map_dt);
    DataType map_from_str = DataTypeUtils::ToType(*map_dt);
    // Expect internalized strings
    EXPECT_EQ(map_dt, map_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(map_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<TestMapToMapInt64ToFloat>()->IsCompatible(from_dt_proto));
  }

  {
    const std::string opaque_map_2("map(int64,opaque(test_domain_2,test_name_2))");
    const auto* map_proto = DataTypeImpl::GetType<MyOpaqueMapCpp_2>()->GetTypeProto();
    EXPECT_NE(map_proto, nullptr);
    DataType map_dt = DataTypeUtils::ToType(*map_proto);
    EXPECT_NE(map_dt, nullptr);
    EXPECT_EQ(opaque_map_2, *map_dt);
    DataType map_from_str = DataTypeUtils::ToType(*map_dt);
    // Expect internalized strings
    EXPECT_EQ(map_dt, map_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(map_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueMapCpp_2>()->IsCompatible(from_dt_proto));
  }

  // Test Sequence with recursion
  {
    const std::string seq_map_str_float("seq(map(string,tensor(float)))");
    const auto* seq_proto = DataTypeImpl::GetType<VectorMapStringToFloat>()->GetTypeProto();
    EXPECT_NE(seq_proto, nullptr);
    DataType seq_dt = DataTypeUtils::ToType(*seq_proto);
    EXPECT_NE(seq_dt, nullptr);
    EXPECT_EQ(seq_map_str_float, *seq_dt);
    DataType seq_from_str = DataTypeUtils::ToType(*seq_dt);
    // Expect internalized strings
    EXPECT_EQ(seq_dt, seq_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(seq_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<VectorMapStringToFloat>()->IsCompatible(from_dt_proto));
  }
#endif

  // Test Sequence with opaque_2
  {
    const std::string seq_opaque_2("seq(opaque(test_domain_2,test_name_2))");
    const auto* seq_proto = DataTypeImpl::GetType<MyOpaqueSeqCpp_2>()->GetTypeProto();
    EXPECT_NE(seq_proto, nullptr);
    DataType seq_dt = DataTypeUtils::ToType(*seq_proto);
    EXPECT_NE(seq_dt, nullptr);
    EXPECT_EQ(seq_opaque_2, *seq_dt);
    DataType seq_from_str = DataTypeUtils::ToType(*seq_dt);
    // Expect internalized strings
    EXPECT_EQ(seq_dt, seq_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(seq_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueSeqCpp_2>()->IsCompatible(from_dt_proto));
  }
  // Test Opaque type opaque_1
  {
    const std::string opaque_q("seq(opaque(test_domain_1,test_name_1))");
    const auto* op_proto = DataTypeImpl::GetType<MyOpaqueSeqCpp_1>()->GetTypeProto();
    EXPECT_NE(op_proto, nullptr);
    DataType op_dt = DataTypeUtils::ToType(*op_proto);
    EXPECT_NE(op_dt, nullptr);
    EXPECT_EQ(opaque_q, *op_dt);
    DataType op_from_str = DataTypeUtils::ToType(*op_dt);
    // Expect internalized strings
    EXPECT_EQ(op_dt, op_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(op_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<MyOpaqueSeqCpp_1>()->IsCompatible(from_dt_proto));
  }
  // Test TestOpaqueDomainOnly
  {
    const std::string opaque_q("opaque(test_domain_1,)");
    const auto* op_proto = DataTypeImpl::GetType<TestOpaqueDomainOnly>()->GetTypeProto();
    EXPECT_NE(op_proto, nullptr);
    DataType op_dt = DataTypeUtils::ToType(*op_proto);
    EXPECT_NE(op_dt, nullptr);
    EXPECT_EQ(opaque_q, *op_dt);
    DataType op_from_str = DataTypeUtils::ToType(*op_dt);
    // Expect internalized strings
    EXPECT_EQ(op_dt, op_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(op_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<TestOpaqueDomainOnly>()->IsCompatible(from_dt_proto));
  }
  // Test TestOpaqueNameOnly
  {
    const std::string opaque_q("opaque(test_name_1)");
    const auto* op_proto = DataTypeImpl::GetType<TestOpaqueNameOnly>()->GetTypeProto();
    EXPECT_NE(op_proto, nullptr);
    DataType op_dt = DataTypeUtils::ToType(*op_proto);
    EXPECT_NE(op_dt, nullptr);
    EXPECT_EQ(opaque_q, *op_dt);
    DataType op_from_str = DataTypeUtils::ToType(*op_dt);
    // Expect internalized strings
    EXPECT_EQ(op_dt, op_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(op_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<TestOpaqueNameOnly>()->IsCompatible(from_dt_proto));
  }
  // Test TestOpaqueNoNames
  {
    const std::string opaque_q("opaque()");
    const auto* op_proto = DataTypeImpl::GetType<TestOpaqueNoNames>()->GetTypeProto();
    EXPECT_NE(op_proto, nullptr);
    DataType op_dt = DataTypeUtils::ToType(*op_proto);
    EXPECT_NE(op_dt, nullptr);
    EXPECT_EQ(opaque_q, *op_dt);
    DataType op_from_str = DataTypeUtils::ToType(*op_dt);
    // Expect internalized strings
    EXPECT_EQ(op_dt, op_from_str);
    const auto& from_dt_proto = DataTypeUtils::ToTypeProto(op_dt);
    EXPECT_TRUE(DataTypeImpl::GetType<TestOpaqueNoNames>()->IsCompatible(from_dt_proto));
  }
}

#ifndef DISABLE_ABSEIL

template <typename T>
using Calc = CalculateInlinedVectorDefaultInlinedElements<T>;

template <typename... Types>
struct TypeMinimunInlinedElements {
  std::array<std::pair<size_t, size_t>, sizeof...(Types)> sizes_{std::make_pair(sizeof(Types), Calc<Types>::value)...};
  void print(std::ostream& os) const {
    os << " CalculateInlinedVectorDefaultInlinedElements Sizes: ";
    for (auto& p : sizes_) {
      os << p.first << " -> " << p.second << std::endl;
    }
    os << std::endl;
  }
};

TEST(InlinedVectorTests, TestDefaultInlinedCapacity) {
  // We want to test all the type here
  TypeMinimunInlinedElements<int8_t, int16_t, int32_t, int64_t, std::string> sizes;
  sizes.print(std::cout);
}

#endif  // ! DISABLE_ABSEIL

TEST(TypeLiterals, Tests) {
  {
    // uint16_t test
    MLFloat16 mlfloat = MLFloat16::FromBits(static_cast<uint16_t>(16));
    auto mlfloat_literal = 16_f16;
    ASSERT_EQ(mlfloat, mlfloat_literal);

    BFloat16 bfloat{static_cast<uint16_t>(16), BFloat16::FromBits()};
    auto bfloat_literal = 16_b16;
    ASSERT_EQ(bfloat, bfloat_literal);
  }

  {
    // float
    MLFloat16 mlfloat{17.0f};
    auto mlfloat_literal = 17.0_fp16;
    ASSERT_EQ(mlfloat, mlfloat_literal);

    BFloat16 bfloat{17.0f};
    auto bfloat_literal = 17.0_bfp16;
    ASSERT_EQ(bfloat, bfloat_literal);
  }
}

}  // namespace test
}  // namespace onnxruntime
