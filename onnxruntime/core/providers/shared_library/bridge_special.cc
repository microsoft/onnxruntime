#include "bridge_special.h"

namespace std {
template <typename T1, typename T2>
struct unordered_map;
}

namespace onnx {
class AttributeProto {
 public:
  AttributeProto();
  AttributeProto(const AttributeProto&);
  virtual ~AttributeProto();
};

AttributeProto::AttributeProto() { onnx_AttributeProto_constructor(this); }
AttributeProto::AttributeProto(const AttributeProto& v1) { onnx_AttributeProto_copy_constructor(this, (void*)&v1); }
AttributeProto::~AttributeProto() { onnx_AttributeProto_destructor(this); }
}  // namespace onnx

namespace onnxruntime {
class TensorShape {
 public:
  TensorShape(__int64 const* p1, unsigned __int64 p2);
};

TensorShape::TensorShape(__int64 const* p1, unsigned __int64 p2) { onnxruntime_TensorShape_constructor(this, p1, p2); }

struct Node;
struct KernelDef;
struct IExecutionProvider;
struct OrtValue {};
struct OrtValueNameIdxMap;
struct FuncManager;
struct DataTransferManager;

class OpKernelInfo {
 public:
  OpKernelInfo::OpKernelInfo(const OpKernelInfo& other);
  OpKernelInfo::OpKernelInfo(const onnxruntime::Node& node,
                             const KernelDef& kernel_def,
                             const IExecutionProvider& execution_provider,
                             const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                             const OrtValueNameIdxMap& ort_value_name_idx_map,
                             const FuncManager& funcs_mgr,
                             const DataTransferManager& data_transfer_mgr);
};

OpKernelInfo::OpKernelInfo(const OpKernelInfo& other) {
  onnxruntime_OpKernelInfo_copy_constructor(this, (void*)&other);
}

OpKernelInfo::OpKernelInfo(const onnxruntime::Node& node,
                           const KernelDef& kernel_def,
                           const IExecutionProvider& execution_provider,
                           const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                           const OrtValueNameIdxMap& ort_value_name_idx_map,
                           const FuncManager& funcs_mgr,
                           const DataTransferManager& data_transfer_mgr) {
  onnxruntime_OpKernelInfo_constructor(this, (void*)&node, (void*)&kernel_def, (void*)&execution_provider, (void*)&constant_initialized_tensors, (void*)&ort_value_name_idx_map, (void*)&funcs_mgr, (void*)&data_transfer_mgr);
};

}  // namespace onnxruntime
