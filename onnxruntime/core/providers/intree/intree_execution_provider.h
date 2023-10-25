#pragma once
#include "interface/provider/provider.h"

namespace onnxruntime {

struct InTreeExecutionProviderInfo {
    int int_property;
    std::string str_property;
};

class InTreeExecutionProvider : public interface::ExecutionProvider {
public:
    InTreeExecutionProvider(const InTreeExecutionProviderInfo& info);
    ~InTreeExecutionProvider() override = default;
private:
    InTreeExecutionProviderInfo info_;
};

class InTreeExecutionProviderFactory {
public:
  InTreeExecutionProviderFactory() {}
  ~InTreeExecutionProviderFactory() {}
  static InTreeExecutionProvider* CreateInTreeExecutionProvider(const std::unordered_map<std::string, std::string>& provider_option);
};

}
