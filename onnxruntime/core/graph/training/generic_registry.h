// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <memory>
#include "core/common/common.h"

namespace onnxruntime {
namespace training {

// GenericRegistry is a helper class for Factory Pattern, e.g. name registration and object creation.
/* Example:
class Base {};
class Derived1 : public Base {};
class Derived2 : public Base {
public: 
  Derived2(int val)
    : val_(val) {}
  int val_;
};

GenericRegistry<Base> reg;

reg.Register<Derived1>("D1");

// Although Derived2 accepts an int parameter, 
// its value must be binded during Register().
// Because reg is with type GenericRegistry<Base>, not GenericRegistry<Base, int>.
reg.Register<Derived2>("D2", []()->unique_ptr<Derived2> {return make_unique<Derived2>(100);});

auto d1 = reg.MakeUnique("D1"); // d1 is an instance of Derived1
auto d2 = reg.MakeUnique("D2"); // d2 is an instance of Derived2, d2->val_ has been always binded to 100.
*/
template <typename BaseType, typename... ConstructorArgTypes>
class GenericRegistry {
 public:
  std::unique_ptr<BaseType> MakeUnique(const std::string& name, ConstructorArgTypes&&... args) const {
    auto it = name_to_creator_func_.find(name);
    if (it != name_to_creator_func_.end()) {
      return (it->second)(std::forward<ConstructorArgTypes>(args)...);
    }
    return nullptr;
  }

  bool Contains(const std::string& name) const {
    return name_to_creator_func_.count(name) != 0;
  }

  typedef std::function<std::unique_ptr<BaseType>(ConstructorArgTypes&&...)> Creator;

  // Register a name using default creator.
  template <typename DerivedType>
  void Register(const std::string& name) {
    ORT_ENFORCE(name_to_creator_func_.count(name) == 0, "Fail to register, the entry exists:", name);
    name_to_creator_func_[name] = [](ConstructorArgTypes&&... args) -> std::unique_ptr<DerivedType> {
      return std::make_unique<DerivedType>(std::forward<ConstructorArgTypes>(args)...);
    };
  }

  // Register a name using custom creator, who can adapt the constructor.
  template <typename DerivedType>
  void Register(const std::string& name, const Creator& creator) {
    ORT_ENFORCE(name_to_creator_func_.count(name) == 0, "Fail to register, the entry exists:", name);
    name_to_creator_func_[name] = creator;
  }

 private:
  std::unordered_map<std::string, Creator> name_to_creator_func_;
};

}  // namespace training
}  // namespace onnxruntime
