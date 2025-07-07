#pragma once

#include "qnn_provider_factory_creator.h"
#include "test/autoep/library/example_plugin_ep_utils.h"

namespace onnxruntime {
class QnnEpFactory;

class QnnEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context{false};
    bool share_ep_contexts{false};
  };

  QnnEp(const QnnEpFactory& factory, const std::string& name,
        const Config& config, const OrtLogger* logger);
  ~QnnEp();

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                  const OrtGraph* graph,
                                                  OrtEpGraphSupportInfo* graph_support_info);

  const QnnEpFactory& factory_;
  std::string name_;
  Config config_;
  const OrtLogger* logger_;
  bool context_cache_enabled_;
  bool share_ep_contexts_;
};

}
