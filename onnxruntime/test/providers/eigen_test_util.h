// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
