// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cloud/endpoint_invoker.h"

namespace onnxruntime {
namespace cloud {

//class AzureMLOp : public OpKernel {
// public:
//  AzureMLOp(const OpKernelInfo& info) : OpKernel(info) {
//    uri_ = info.GetAttrOrDefault<std::string>("uri", "");
//    key_ = info.GetAttrOrDefault<std::string>("key", "");
//    type_ = EndPointInvoker::MapType(info.GetAttrOrDefault<std::string>("type", "rest"));
//    EndPointConfig config;
//    config["uri"] = uri_;
//    config["key"] = key_;
//    invoker_ = EndPointInvoker::CreateInvoker(type_, config);
//  }
//
//  common::Status Compute(OpKernelContext* context) const override;
//
// private:
//  std::string uri_;
//  std::string key_;
//  EndPointType type_;
//  std::unique_ptr<EndPointInvoker> invoker_;
//};

}  // namespace cloud
}  // namespace onnxruntime