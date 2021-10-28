// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

class NodeArg;
class Path;

class INodeUnit {
 public:
  virtual ~INodeUnit() = default;

  virtual const ConstPointerContainer<std::vector<NodeArg*>> InputDefs() const noexcept = 0;
  virtual const ConstPointerContainer<std::vector<NodeArg*>> OutputDefs() const noexcept = 0;

  virtual const std::string& OpType() const noexcept = 0;
  virtual int SinceVersion() const noexcept = 0;
  virtual const std::string& Domain() const noexcept = 0;
  virtual const Path& ModelPath() const noexcept = 0;
  virtual const std::string& Name() const noexcept = 0;

  virtual const Node& GetNode() const noexcept = 0;
};

const std::unique_ptr<INodeUnit> CreateNodeUnit(const Node& node);

}  // namespace onnxruntime
