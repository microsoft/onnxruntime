#include "bridge_special.h"

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

}  // namespace onnxruntime
