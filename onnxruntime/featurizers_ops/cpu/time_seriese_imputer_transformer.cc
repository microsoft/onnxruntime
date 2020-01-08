// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/TimeSeriesImputerFeaturizer.h"
#include "Archive.h"

namespace ft = Microsoft::Featurizer::Featurizers;

namespace onnxruntime {
namespace featurizers {

// Double and float
template <typename T>
inline nonstd::optional<std::string> PreprocessOptional(T value) { 
  nonstd::optional<std::string> result;
  if (std::isnan(value)) {
    return result;
  }
  result = std::to_string(value);
  return result;
}

inline nonstd::optional<std::string> PreprocessOptional(const std::string& value) {
  return value.empty() ? nonstd::optional<std::string>() : nonstd::optional<std::string>(value);
}

template <typename T>
struct TimeSeriesImputerTransformerImpl {
  Status operator()(OpKernelContext* ctx, int64_t batches, int64_t rows) {
    const auto& times = *ctx->Input<Tensor>(1);
    const auto& keys = *ctx->Input<Tensor>(2);
    const auto& data = *ctx->Input<Tensor>(2);

    const bool explicit_batch = data.Shape().NumDimensions() == 3;
    const int64_t keys_per_row = (keys.Shape().NumDimensions() == 2) ? keys.Shape()[1] : keys.Shape()[2];
    const int64_t columns = (data.Shape().NumDimensions() == 2) ? data.Shape()[1] : data.Shape()[2];

    using OutputType = std::tuple<bool, std::chrono::system_clock::time_point, std::string, nonstd::optional<std::string>>;
    std::vector<std::vector<OutputType>> output_batches;

    for (int64_t batch = 0; batch < batches; ++batch) {
      const int64_t* times_data = times.template Data<int64_t>() + batch * rows;
      const T* keys_data = keys.template Data<T>() + batch * rows * keys_per_row;
      const T* data_data = data.template Data<T>() + batch * rows * columns;

       // for each row get timestamp, get all keys, get all data and feed it
      for (int64_t row = 0; row < rows; ++row) {
        keys_data = keys_data + (row * keys_per_row);
        const T* const keys_data_end = keys_data + keys_per_row;
        std::vector<std::string> str_keys;
        std::transform(keys_data, keys_data_end, std::back_inserter(str_keys), PreprocessOptional);

        std::vector<std::string> str_data;
        data_data = data_data + (row * columns);
        const T* const data_end = data_data + columns;
        std::transform(data_data, data_end, std::back_inserter(str_data), PreprocessOptional);
        auto tuple_row = std::make_tuple(*times_data, std::move(str_keys), std::move(str_data));

        std::vector<OutputType> output;
        auto const callback(
            [&output](OutputType value) {
              output.emplace_back(std::move(value));
            });
      }
      // and create a vector of rows (InputType)
    }
  }
};

class TimeSeriesImputerTransformer final : public OpKernel {
 public:
  explicit TimeSeriesImputerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  static Status CheckBatches(int64_t batches, int64_t rows, const TensorShape& shape) {
    if (shape.NumDimensions() == 2) {
      ORT_RETURN_IF_NOT(batches == 1, "Number of batches does not match");
      ORT_RETURN_IF_NOT(rows == shape[0], "Number of rows does not match");
    } else if (shape.NumDimensions() == 3) {
      ORT_RETURN_IF_NOT(batches == shape[0], "Number of batches does not match");
      ORT_RETURN_IF_NOT(rows == shape[1], "Number of rows does not match");
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expect shape of [B][R][C] or [R][C]");
    }
    return Status::OK();
  }

  Status Compute(OpKernelContext* ctx) const override {
    const auto& times = *ctx->Input<Tensor>(1);
    const auto& times_shape = times.Shape();
    int64_t batches = 1;
    int64_t rows = 0;
    if (times_shape.NumDimensions() == 2) {
      batches = times_shape[0];
      rows = times_shape[1];
    } else {
      ORT_RETURN_IF_NOT(times_shape.NumDimensions() == 1, "Times must have shape [B][R] or [R]");
      rows = times_shape[0];
    }

    const auto& keys = *ctx->Input<Tensor>(2);
    ORT_RETURN_IF_ERROR(CheckBatches(batches, rows, keys.Shape()));
    const auto& data = *ctx->Input<Tensor>(2);
    ORT_RETURN_IF_ERROR(CheckBatches(batches, rows, data.Shape()));

    auto data_type = data.GetElementType();
    ORT_RETURN_IF_NOT(keys.GetElementType() == data_type, "Keys and data must have the same datatype");

    //utils::MLTypeCallDispatcher<TimeSeriesImputerTransformerImpl, float, double, std::string> t_disp(data_type);
    //t_disp.Invoke(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    TimeSeriesImputerTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<double>(),
                               DataTypeImpl::GetTensorType<std::string>()}),
    TimeSeriesImputerTransformer);
}  // namespace featurizers
}  // namespace onnxruntime
