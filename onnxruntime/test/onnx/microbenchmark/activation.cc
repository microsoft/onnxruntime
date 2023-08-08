#include "common.h"

#include "core/common/status.h"
#include "core/session/ort_env.h"
#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/data_transfer_manager.h"
#include "core/util/thread_utils.h"
#include "core/framework/node_index_info.h"
#include "core/framework/execution_frame.h"
#include "contrib_ops/cpu/activations.h"
#include "core/providers/cpu/activation/activations.h"
#include <onnx/defs/attr_proto_util.h>
#include <benchmark/benchmark.h>
#include <random>

using namespace onnxruntime::common;
using namespace onnxruntime;
using namespace onnx;
extern OrtEnv* env;

class Allocs : public IExecutionProvider {
 private:
  std::shared_ptr<CPUAllocator> alloc = std::make_shared<CPUAllocator>();

 public:
  Allocs() : IExecutionProvider("fake"){};
  AllocatorPtr GetAllocator(OrtMemType) const {
    return alloc;
  }
};

struct KernelAndDef {
  std::unique_ptr<KernelDef> def;
  std::unique_ptr<Model> model;
  std::unique_ptr<logging::Logger> test_logger;
  std::unique_ptr<OpKernel> kernel;
  std::unique_ptr<OrtValueNameIdxMap> ort_value_idx_map = std::make_unique<OrtValueNameIdxMap>();
  std::unique_ptr<Allocs> a = std::make_unique<Allocs>();

  template <typename KernelType>
  static KernelAndDef CreateKernel(const std::string& op_name, const std::string& domain,
                                   const std::vector<AttributeProto>& attrs, int64_t batch_size) {
    std::unordered_map<std::string, int> domain2Version;
    domain2Version[""] = 12;
    domain2Version[kMSDomain] = 1;
    KernelAndDef out;
    out.test_logger = env->GetLoggingManager()->CreateLogger("test");
    out.model = std::make_unique<Model>("graph_1", false, *out.test_logger);
    auto& graph = out.model->MainGraph();
    TypeProto tensor_float;
    tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(batch_size);
    auto& input_arg = graph.GetOrCreateNodeArg("input", &tensor_float);
    auto& output_arg = graph.GetOrCreateNodeArg("output", &tensor_float);
    out.ort_value_idx_map->Add("input");
    out.ort_value_idx_map->Add("output");
    std::unordered_map<std::string, AttributeProto> attributes;
    for (const AttributeProto& p : attrs) {
      attributes[p.name()] = p;
    }
    Node& main_node = graph.AddNode("main", op_name, "", {&input_arg}, {&output_arg}, &attributes, domain);
    ORT_THROW_IF_ERROR(graph.Resolve());
    main_node.SetExecutionProviderType("fake");
    out.def = KernelDefBuilder()
                  .SetName(op_name)
                  .SetDomain(domain)
                  .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                  .Build();
    OpKernelInfo info(main_node, *out.def, *out.a, {}, {}, {});
    out.kernel = std::make_unique<KernelType>(info);
    return out;
  }
};

class MyIExecutionFrame : public IExecutionFrame {
 private:
  IExecutionProvider& a_;

 public:
  MyIExecutionFrame(IExecutionProvider& a, const std::vector<int>& feed_mlvalue_idxs,
                    const std::vector<OrtValue>& feeds, const std::unordered_map<int, OrtValue>& initializers,
                    const std::vector<int>& fetch_mlvalue_idxs, const std::vector<OrtValue>& fetches,
                    const OrtValueNameIdxMap& ort_value_idx_map, const NodeIndexInfo& node_index_info)
      : IExecutionFrame(ort_value_idx_map, node_index_info, fetch_mlvalue_idxs),
        a_(a) {
    Init(
        feed_mlvalue_idxs, feeds, initializers, [](const std::string& /*name*/) -> bool { return false; }, fetches);
  }

  const DataTransferManager& GetDataTransferManager() const override {
    abort();
  }

  Status CreateNodeOutputMLValueImpl(OrtValue& /*ort_value*/, int /*ort_value_idx*/, const TensorShape* /*shape*/) override {
    abort();
  }
  AllocatorPtr GetAllocatorImpl(const OrtDevice&) const override {
    return static_cast<Allocs&>(a_).GetAllocator(OrtMemTypeDefault);
  }

  Status CreateNodeOutputMLValueImpl(OrtValue& ort_value, int ort_value_index, const TensorShape* shape, size_t) {
    using T = float;
    if (ort_value_index == NodeIndexInfo::kInvalidEntry) {
      return Status(ONNXRUNTIME, FAIL, "Trying to allocate memory for unused optional inputs/outputs");
    }
    size_t size;
    int64_t len = shape->Size();
    if (len < 0) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape cannot contain any negative value");
    }
    if (static_cast<uint64_t>(len) > std::numeric_limits<size_t>::max()) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Tensor shape is too large");
    }

    if (!IAllocator::CalcMemSizeForArrayWithAlignment<0>(static_cast<size_t>(len), sizeof(T), &size)) {
      return Status(ONNXRUNTIME, FAIL, "size overflow");
    }
    auto alloc = static_cast<Allocs&>(a_).GetAllocator(OrtMemTypeDefault);
    std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<T>(), *shape, alloc);

    auto ml_tensor = DataTypeImpl::GetType<Tensor>();
    ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
    return Status::OK();
  }

  Status CopyTensor(const Tensor& /*src*/, Tensor& /*dest*/) const override {
    return Status::OK();
  }
};

template <typename KernelType>
static void RunSingleNode(const std::string& op_name, const std::string& domain,
                          const std::vector<AttributeProto>& attrs, benchmark::State& state, float low = -1.0f,
                          float high = 1.0f) {
  const int64_t batch_size = state.range(0);
  float* output = (float*)aligned_alloc(sizeof(float) * static_cast<size_t>(batch_size), 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size, low, high);
  KernelAndDef k = KernelAndDef::CreateKernel<KernelType>(op_name, domain, attrs, batch_size);

  std::vector<int> feed_mlvalue_idxs(1);
  std::vector<int> fetch_mlvalue_idxs(1);
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("input", feed_mlvalue_idxs[0]));
  ORT_THROW_IF_ERROR(k.ort_value_idx_map->GetIdx("output", fetch_mlvalue_idxs[0]));

  std::vector<OrtValue> feeds(1);
  std::vector<OrtValue> fetches(1);
  TensorShapeVector shapes(static_cast<size_t>(1), batch_size);
  OrtMemoryInfo info("cpu", OrtDeviceAllocator);
  auto ml_float = DataTypeImpl::GetType<float>();
  Tensor::InitOrtValue(ml_float, shapes, data, info, feeds[0]);
  Tensor::InitOrtValue(ml_float, shapes, output, info, fetches[0]);
  GraphViewer v(k.model->MainGraph());
  NodeIndexInfo node_index_info(v, *k.ort_value_idx_map);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  MyIExecutionFrame f(*k.a, feed_mlvalue_idxs, feeds, {}, fetch_mlvalue_idxs, fetches, *k.ort_value_idx_map,
                      node_index_info);
  for (auto _ : state) {
    OpKernelContext c(&f, k.kernel.get(), /*stream*/ nullptr, tp.get(), *k.test_logger);
    Status st = k.kernel->Compute(&c);
    if (!st.IsOK())
      state.SkipWithError(st.ErrorMessage().c_str());
  }
  aligned_free(data);
  aligned_free(output);
}

static void BM_GeluCompute(benchmark::State& state) {
  RunSingleNode<contrib::Gelu<float>>("Gelu", kMSDomain, {}, state);
}

BENCHMARK(BM_GeluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(98304)
    ->Arg(1572864);

static void BM_ScaledTanhCompute(benchmark::State& state) {
  RunSingleNode<contrib::ScaledTanh<float>>("ScaledTanh", kMSDomain,
                                            {MakeAttribute("alpha", 0.8f), MakeAttribute("beta", 0.3f)}, state);
}

BENCHMARK(BM_ScaledTanhCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);

static void BM_EluCompute(benchmark::State& state) {
  RunSingleNode<Elu<float>>("Elu", "",
                            {
                                MakeAttribute("alpha", 0.8f),
                            },
                            state);
}

BENCHMARK(BM_EluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(2000)
    ->Arg(4000)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000);

static void BM_HardSigmoidCompute(benchmark::State& state) {
  RunSingleNode<HardSigmoid<float>>("HardSigmoid", "", {MakeAttribute("alpha", 0.2f), MakeAttribute("beta", 0.5f)},
                                    state, 0.1f, 0.6f);
}

BENCHMARK(BM_HardSigmoidCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_LeakyReluCompute(benchmark::State& state) {
  RunSingleNode<LeakyRelu<float>>("LeakyRelu", "", {MakeAttribute("alpha", 0.2f)}, state);
}

BENCHMARK(BM_LeakyReluCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(4000)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000);

static void BM_SoftplusCompute(benchmark::State& state) {
  RunSingleNode<Softplus<float>>("Softplus", "", {}, state, -2.0f, 2.0f);
}

BENCHMARK(BM_SoftplusCompute)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Selu(benchmark::State& state) {
  RunSingleNode<Selu<float>>("Selu", "", {}, state, -2.0f, 2.0f);
}

BENCHMARK(BM_Selu)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Sigmoid(benchmark::State& state) {
  RunSingleNode<Sigmoid<float>>("Sigmoid", "", {}, state, -2.0f, 2.0f);
}

BENCHMARK(BM_Sigmoid)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Softsign(benchmark::State& state) {
  RunSingleNode<Softsign<float>>("Softsign", "", {}, state, -2.0f, 2.0f);
}

BENCHMARK(BM_Softsign)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Tanh(benchmark::State& state) {
  RunSingleNode<Tanh<float>>("Tanh", "", {}, state, -2.0f, 2.0f);
}

BENCHMARK(BM_Tanh)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

static void BM_Relu(benchmark::State& state) {
  RunSingleNode<Relu<float>>("Relu", "", {}, state, -2.0f, 2.0f);
}

BENCHMARK(BM_Relu)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Arg(40000)
    ->Arg(80000)
    ->Arg(160000)
    ->Arg(320000)
    ->Arg(640000)
    ->Arg(1280000);

template <typename T>
struct Powx {
  const T* input1 = nullptr;
  const T* input2 = nullptr;
  T* output = nullptr;
  float Cost() const {
    return 30.f;
  }
  void operator()(std::ptrdiff_t first, std::ptrdiff_t last) const {
    ptrdiff_t len = last - first;
    T* output_ptr = this->output + first;
    const T* in1 = this->input1 + first;
    const T* in2 = this->input2 + first;
    for (ptrdiff_t i = 0; i != len; ++i) {
      output_ptr[i] = std::pow(in1[i], in2[i]);
    }
  }
};

static void BM_Powx(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const int cost = static_cast<int>(state.range(1));
  float* output = (float*)aligned_alloc(sizeof(float) * batch_size, 64);
  float* input2 = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  float* input1 = GenerateArrayWithRandomValue<float>(batch_size, -1, 1);
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));
  Powx<float> f;
  f.input1 = input1;
  f.input2 = input2;
  f.output = output;
  for (auto _ : state) {
    concurrency::ThreadPool::TryParallelFor(tp.get(), batch_size, TensorOpCost{2, 1, static_cast<double>(cost)}, f);
  }
  aligned_free(input1);
  aligned_free(input2);
  aligned_free(output);
}

BENCHMARK(BM_Powx)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kNanosecond)
    ->Args({100, 1})
    ->Args({100, 5})
    ->Args({100, 10})
    ->Args({100, 40})
    ->Args({100, 80})
    ->Args({100, 160})
    ->Args({100, 320})
    ->Args({100, 640})
    ->Args({500, 1})
    ->Args({500, 5})
    ->Args({500, 10})
    ->Args({500, 40})
    ->Args({500, 80})
    ->Args({500, 160})
    ->Args({500, 320})
    ->Args({500, 640})
    ->Args({1000, 1})
    ->Args({1000, 5})
    ->Args({1000, 10})
    ->Args({1000, 40})
    ->Args({1000, 80})
    ->Args({1000, 160})
    ->Args({1000, 320})
    ->Args({2000, 1})
    ->Args({2000, 5})
    ->Args({2000, 10})
    ->Args({2000, 40})
    ->Args({2000, 80})
    ->Args({2000, 160})
    ->Args({2000, 320})
    ->Args({2500, 1})
    ->Args({2500, 5})
    ->Args({2500, 10})
    ->Args({2500, 40})
    ->Args({2500, 80})
    ->Args({2500, 160})
    ->Args({5000, 1})
    ->Args({5000, 5})
    ->Args({5000, 10})
    ->Args({5000, 40})
    ->Args({5000, 80})
    ->Args({5000, 160});
