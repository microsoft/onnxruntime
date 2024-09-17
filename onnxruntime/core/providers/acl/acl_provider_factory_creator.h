// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {

struct ACLProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(bool enable_fast_math);
};

}  // namespace onnxruntime
