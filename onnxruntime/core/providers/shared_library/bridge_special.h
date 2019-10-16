void onnx_AttributeProto_constructor(void* _this);
void onnx_AttributeProto_copy_constructor(void* _this, void* copy);
void onnx_AttributeProto_destructor(void* _this);

void onnxruntime_Status_constructor_1(void* _this, const void* category, int code, char const* msg);
void onnxruntime_Status_constructor_2(void* _this, const void* category, int code, const void* std_string_msg);

void onnxruntime_TensorShape_constructor(void* _this, __int64 const* p1, unsigned __int64 p2);

void onnxruntime_OpKernelInfo_constructor(void* _this, void* p1, void* p2, void* p3, void* p4, void* p5, void* p6, void* p7);
void onnxruntime_OpKernelInfo_copy_constructor(void* _this, void* copy);
