#pragma once
#include "core/common/status.h"
#include "core/framework/execution_frame.h"
#include "core/framework/execution_provider.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/node_index_info.h"
#include "core/framework/kernel_def_builder.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/util/thread_utils.h"
#include <benchmark/benchmark.h>

class FakeEP : public onnxruntime::IExecutionProvider {
 private:
  std::shared_ptr<onnxruntime::CPUAllocator> alloc = std::make_shared<onnxruntime::CPUAllocator>();

 public:
  FakeEP() : onnxruntime::IExecutionProvider("fake"){};
  virtual onnxruntime::AllocatorPtr GetAllocator(int, OrtMemType) const {
    return alloc;
  }
};

// A dummy ExecutionFrame that assumes every tensor is in float
class MyIExecutionFrame : public onnxruntime::IExecutionFrame {
 private:
  onnxruntime::IExecutionProvider& a_;

 public:
  MyIExecutionFrame(onnxruntime::IExecutionProvider& a, const std::vector<int>& feed_mlvalue_idxs,
                    const std::vector<OrtValue>& feeds, const std::unordered_map<int, OrtValue>& initializers,
                    const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                    const onnxruntime::OrtValueNameIdxMap& ort_value_idx_map,
                    const onnxruntime::NodeIndexInfo& node_index_info)
      : IExecutionFrame(feed_mlvalue_idxs, feeds, initializers, fetch_mlvalue_idxs, fetches, ort_value_idx_map,
                        node_index_info),
        a_(a) {
  }

  onnxruntime::AllocatorPtr GetAllocatorImpl(const OrtMemoryInfo& info) const {
    return a_.GetAllocator(info.id, info.mem_type);
  }

  onnxruntime::Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_index,
                                                  const onnxruntime::TensorShape* shape, size_t);
};

struct KernelAndDef {
  std::unique_ptr<onnxruntime::KernelDef> def;
  std::unique_ptr<onnxruntime::Model> model;
  std::unique_ptr<onnxruntime::logging::Logger> test_logger;
  std::unique_ptr<onnxruntime::OpKernel> kernel;
  std::unique_ptr<onnxruntime::OrtValueNameIdxMap> ort_value_idx_map =
      std::make_unique<onnxruntime::OrtValueNameIdxMap>();
  std::unique_ptr<FakeEP> a = std::make_unique<FakeEP>();

  template <typename KernelType>
  static KernelAndDef CreateKernel(const std::string& op_name, const std::string& domain,
                                   const std::vector<onnx::AttributeProto>& attrs, int64_t batch_size) {
    std::vector<int64_t> v;
    v.push_back(batch_size);
    return CreateKernel<KernelType>(op_name, domain, attrs, v);
  }
  template <typename KernelType>
  static KernelAndDef CreateKernel(const std::string& op_name, const std::string& domain,
                                   const std::vector<onnx::AttributeProto>& attrs,
                                   const std::vector<int64_t>& input_shape) {
    std::unordered_map<std::string, int> domain2Version;
    domain2Version[""] = 12;
    domain2Version[kMSDomain] = 1;
    KernelAndDef out;
    out.test_logger = env->GetLoggingManager()->CreateLogger("test");
    out.model = std::make_unique<onnxruntime::Model>("graph_1", false, *out.test_logger);
    auto& graph = out.model->MainGraph();
    TypeProto tensor_float;
    tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto* p = tensor_float.mutable_tensor_type()->mutable_shape();
    for (auto i : input_shape) {
      p->add_dim()->set_dim_value(i);
    }
    auto& input_arg = graph.GetOrCreateNodeArg("input", &tensor_float);
    TypeProto tensor_float2;
    tensor_float2.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    auto& output_arg = graph.GetOrCreateNodeArg("output", &tensor_float2);
    out.ort_value_idx_map->Add("input");
    out.ort_value_idx_map->Add("output");
    std::unordered_map<std::string, onnx::AttributeProto> attributes;
    for (const AttributeProto& proto : attrs) {
      attributes[proto.name()] = proto;
    }
    onnxruntime::Node& main_node = graph.AddNode("main", op_name, "", {&input_arg}, {&output_arg}, &attributes, domain);
    ORT_THROW_IF_ERROR(graph.Resolve());
    main_node.SetExecutionProviderType("fake");
    out.def = KernelDefBuilder()
                  .SetName(op_name)
                  .SetDomain(domain)
                  .TypeConstraint("T", onnxruntime::DataTypeImpl::GetTensorType<float>())
                  .Build();
    onnxruntime::OpKernelInfo info(main_node, *out.def, *out.a, {}, {}, {}, {});
    out.kernel = std::make_unique<KernelType>(info);
    return out;
  }
};

template <typename KernelType>
static void RunSingleNode(const std::string& op_name, const std::string& domain,
                          const std::vector<onnx::AttributeProto>& attrs, OrtThreadPoolImplType impl_type,
                          benchmark::State& state, float low = -1.0f, float high = 1.0f) {
  const int64_t batch_size = state.range(0);
  float* output = (float*)_aligned_malloc(sizeof(float) * static_cast<size_t>(batch_size), 64);
  float* data = GenerateFloatArray(batch_size, low, high);
  KernelAndDef k = KernelAndDef::CreateKernel<KernelType>(op_name, domain, attrs, batch_size);

  std::vector<int> feed_mlvalue_idxs(1);
  std::vector<int> fetch_mlvalue_idxs(1);
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("input", feed_mlvalue_idxs[0]));
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("output", fetch_mlvalue_idxs[0]));

  std::vector<OrtValue> feeds(1);
  std::vector<OrtValue> fetches(1);
  std::vector<int64_t> shapes(static_cast<size_t>(1), batch_size);
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  OrtMemoryInfo info("cpu", OrtDeviceAllocator);
  feeds[0].Init(new Tensor(DataTypeImpl::GetType<float>(), shapes, data, info), ml_tensor, ml_tensor->GetDeleteFunc());
  fetches[0].Init(new Tensor(DataTypeImpl::GetType<float>(), shapes, output, info), ml_tensor,
                  ml_tensor->GetDeleteFunc());
  GraphViewer v(k.model->MainGraph());
  NodeIndexInfo node_index_info(v, *k.ort_value_idx_map);
  OrtThreadPoolParams tpo;
  tpo.impl_type = impl_type;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  MyIExecutionFrame f(*k.a, feed_mlvalue_idxs, feeds, {}, fetch_mlvalue_idxs, fetches, *k.ort_value_idx_map,
                      node_index_info);
  for (auto _ : state) {
    OpKernelContext c(&f, k.kernel.get(), tp.get(), *k.test_logger);
    Status st = k.kernel->Compute(&c);
    if (!st.IsOK())
      state.SkipWithError(st.ErrorMessage().c_str());
  }
  _aligned_free(data);
  _aligned_free(output);
}
