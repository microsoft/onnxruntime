// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "interface/provider/provider.h"

namespace onnxruntime {

namespace test {

struct CustomEpInfo {
  int int_property;
  std::string str_property;
};

class CustomEp : public interface::ExecutionProvider {
 public:
  CustomEp(const CustomEpInfo& info);
  ~CustomEp() override = default;

  bool CanCopy(const OrtDevice& src, const OrtDevice& dest) override;
  //void MemoryCpy(Ort::UnownedValue&, Ort::ConstValue const&) override;

  std::vector<std::unique_ptr<interface::SubGraphDef>> GetCapability(interface::GraphViewRef*) override;
  void RegisterKernels(interface::IKernelRegistry&) override;

  common::Status Compile(std::vector<std::unique_ptr<interface::GraphViewRef>>&,
                         std::vector<std::unique_ptr<interface::NodeViewRef>>&,
                         std::vector<NodeComputeInfo>&) override;

 private:
  CustomEpInfo info_;
};

}  // namespace test

}  // namespace onnxruntime