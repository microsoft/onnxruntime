// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"

// TODO retire this file

namespace onnxruntime {

// A mini IR layer for shape inference
// Currently just use tvm::Expr but can be replaced by others later
// Following features are needed:
// 1. represent symbolic int
// 2. represent +-*/
// 3. check if two DimExpr is the same
// 4. simplify if needed
// For now only symbolic int is supported
class SimpleDimExpr {
 public:
  SimpleDimExpr() : has_value_(false) {}
  SimpleDimExpr(int64_t i) : value_(i), has_value_(true) {}
  SimpleDimExpr(const std::string& sym) : symbol_(sym), has_value_(false) {}
  bool IsConst() const { return has_value_; }
  bool IsOne() const { return has_value_ && value_ == 1; }
  bool operator==(const SimpleDimExpr& expr) const {
    if (has_value_ != expr.has_value_)
      return false;

    if (has_value_)
      return value_ == expr.value_;
    else
      return symbol_ == expr.symbol_;
  }

  bool operator!=(const SimpleDimExpr& expr) const {
    return !(*this == expr);
  }

  SimpleDimExpr operator+(const SimpleDimExpr& other) const {
    ORT_ENFORCE(has_value_ && other.has_value_);
    return SimpleDimExpr(value_ + other.value_);
  }

  SimpleDimExpr operator-(const SimpleDimExpr& other) const {
    ORT_ENFORCE(has_value_ && other.has_value_);
    return SimpleDimExpr(value_ - other.value_);
  }

  SimpleDimExpr operator*(const SimpleDimExpr& other) const {
    if (has_value_ && other.has_value_)
      return SimpleDimExpr(value_ * other.value_);
    else if (IsOne())
      return other;
    else if (other.IsOne())
      return *this;
    else
      ORT_ENFORCE(false, "unsupported symbolic dim computation");
  }

  SimpleDimExpr operator/(const SimpleDimExpr& other) const {
    if (has_value_ && other.has_value_)
      return SimpleDimExpr(value_ / other.value_);
    else if (other.IsOne())
      return *this;
    else
      ORT_ENFORCE(false, "unsupported symbolic dim computation");
  }

  SimpleDimExpr operator%(const SimpleDimExpr& other) const {
    ORT_ENFORCE(has_value_ && other.has_value_);
    return SimpleDimExpr(value_ % other.value_);
  }

  int64_t Value() const {
    ORT_ENFORCE(IsConst());
    return value_;
  }

  const std::string& Symbol() const {
    ORT_ENFORCE(!IsConst());
    return symbol_;
  }

  std::string ToString() const {
    if (has_value_)
      return std::to_string(value_);
    else
      return symbol_;
  }

 private:
  std::string symbol_;
  int64_t value_;
  bool has_value_;
};

template <typename DimT>
class ShapeExprT {
 public:
  ShapeExprT() = default;
  ShapeExprT(const ShapeExprT<DimT>& expr) = default;
  ShapeExprT(ShapeExprT<DimT>&& expr) = default;
  ShapeExprT(size_t size) { dims_.resize(size); }
  ShapeExprT(const std::vector<DimT>& dims) : dims_(dims) {}
  ShapeExprT(const std::vector<int64_t>& dims) {
    for (auto dim : dims)
      dims_.push_back(DimT(dim));
  }

  size_t Rank() const {
    return dims_.size();
  }

  int64_t TotalKnown() const {
    if (dims_.size() == 0)
      return 1;
    int64_t total = 1;
    for (size_t i = 0; i < dims_.size(); ++i) {
      if (dims_[i].IsConst())
        total = total * dims_[i].Value();
    }
    return total;
  }

  size_t KnownFromDimension() const {
    size_t min_index = dims_.size();
    for (int i = static_cast<int>(dims_.size() - 1); i >= 0; i--) {
      if (!dims_[i].IsConst())
        break;
      min_index = static_cast<size_t>(i);
    }
    return min_index;
  }

  std::vector<int64_t> TailedKnown() const {
    std::vector<int64_t> result;

    for (size_t i = KnownFromDimension(); i < Rank(); ++i) {
      result.push_back(dims_[i].Value());
    }
    return result;
  }

  int64_t TotalTailedKnown() const {
    int64_t result = 1;
    for (size_t i = KnownFromDimension(); i < Rank(); ++i) {
      result *= dims_[i].Value();
    }
    return result;
  }

  /**
  Return the total number of elements up to the specified dimension.
  @param dim Return size up to this dimension. Value must be >= 0 and < this.Size().
  */
  DimT SizeToDimension(size_t dim) const {
    DimT total(1);
    for (size_t i = 0; i < std::min(dim, dims_.size()); ++i)
      total = total * dims_[i];
    return total;
  }

  /**
  Return the total number of elements from the specified dimension to the end of the tensor shape.
  @param dim Return size up to this dimension. 0 <= dimension < this.Size().
  */
  DimT SizeFromDimension(size_t dim) const {
    DimT total(1);
    for (size_t i = dim; i < dims_.size(); ++i)
      total = total * dims_[i];
    return total;
  }

  bool IsConst() const {
    return std::all_of(dims_.begin(), dims_.end(), [](const DimT& dim) { return dim.IsConst(); });
  }

  bool operator==(const ShapeExprT<DimT>& shape) const {
    if (Rank() != shape.Rank())
      return false;

    for (size_t dim = 0; dim < Rank(); ++dim) {
      if (dims_[dim] != (shape.dims_[dim]))
        return false;
    }
    return true;
  }

  const ShapeExprT<DimT>& operator=(const ShapeExprT<DimT>& shape) {
    dims_ = shape.dims_;
    return *this;
  }

  const DimT& at(size_t dim) const {
    ORT_ENFORCE(dim < Rank());
    return dims_[dim];
  }

  DimT& at(size_t dim) {
    ORT_ENFORCE(dim < Rank());
    return dims_[dim];
  }

  const DimT& operator[](size_t dim) const {
    ORT_ENFORCE(dim < Rank());
    return dims_[dim];
  }

  DimT& operator[](size_t dim) {
    ORT_ENFORCE(dim < Rank());
    return dims_[dim];
  }

  const std::vector<int64_t> Value() const {
    ORT_ENFORCE(IsConst());
    std::vector<int64_t> result;
    for (size_t i = 0; i < Rank(); ++i) {
      result.push_back(dims_[i].Value());
    }
    return result;
  }

  std::string ToString() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < Rank(); ++i) {
      if (i > 0)
        oss << ", ";
      oss << dims_[i].ToString();
    }
    oss << ")";
    return oss.str();
  }

 private:
  std::vector<DimT> dims_;
};

typedef SimpleDimExpr DimExpr;
typedef ShapeExprT<DimExpr> ShapeExpr;

}  // namespace onnxruntime
