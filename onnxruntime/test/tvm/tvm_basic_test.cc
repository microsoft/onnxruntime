// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include <tvm/runtime/ndarray.h>
#include "core/codegen/tvm/tvm_kernel.h"
#include "core/framework/execution_provider.h"
#include "core/framework/computation_capacity.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/common/logging/logging.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

tvm::Schedule DefaultTVMScheduleGenerator(const TVMGraph& tvm_graph) {
  std::vector<tvm::Operation> args;
  for (auto& tensor : tvm_graph.outputs_)
    args.push_back(tensor.tvm_tensor_->op);
  return tvm::create_schedule(args);
}

tvm::runtime::Module BuildStackVMDefaultModule(tvm::Schedule schedule, tvm::BuildConfig config, tvm::Array<tvm::Tensor> tvm_args, std::vector<std::string>& target_func_names) {
  auto target = tvm::target::stackvm();
  std::string func_name = "func";
  auto args = tvm::Array<tvm::Tensor>(tvm_args);
  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
  auto lowered = lower(schedule, args, "func", binds, config);
  target_func_names.push_back(func_name);
  return build(lowered, target, tvm::Target(), config);
}

template <TVMScheduleCreator S, TVMModuleBuilder M>
class TVMFuseAddKernels : public TVMKernel<S, M> {
 public:
  explicit TVMFuseAddKernels(const OpKernelInfo& info) : TVMKernel<S, M>(info) {}

 protected:
  virtual const TensorShape& GetOutputShape(OpKernelContext* context, int /*i*/) const override {
    return context->Input<Tensor>(0)->Shape();
  }
};

class UnionSet {
 public:
  UnionSet(int n) {
    for (int i = 0; i < n; ++i) {
      farthers_.push_back(i);
    }
  }

  int get(int x) {
    if (farthers_[x] == x) {
      return x;
    }
    return farthers_[x] = get(farthers_[x]);
  }

  void merge(int x, int y) {
    x = get(x);
    y = get(y);
    if (x != y) {
      farthers_[y] = x;
    }
  }

  std::vector<int> farthers_;
};

void FuseAdd(const onnxruntime::GraphViewer& graph, std::vector<std::unique_ptr<ComputationCapacity>>& capacities) {
  std::vector<onnxruntime::NodeIndex> add_nodes;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Add") {
      add_nodes.push_back(node.Index());
    }
  }

  UnionSet set(static_cast<int>(add_nodes.size()));
  for (int i = 0; i < add_nodes.size(); ++i) {
    auto node = graph.GetNode(add_nodes[i]);
    for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
      auto index_it = std::find(add_nodes.begin(), add_nodes.end(), (*it).Index());
      if (index_it != add_nodes.end()) {
        set.merge(i, static_cast<int>(index_it - add_nodes.begin()));
      }
    }
  }

  std::vector<std::vector<onnxruntime::NodeIndex>> groups;
  groups.resize(add_nodes.size());
  for (int i = 0; i < set.farthers_.size(); ++i) {
    groups[set.get(i)].push_back(add_nodes[i]);
  }

  for (auto& group : groups) {
    if (group.size() > 1) {
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      std::set<const onnxruntime::NodeArg*> fused_inputs, fused_outputs;
      for (auto index : group) {
        sub_graph->nodes.push_back(index);
        auto node = graph.GetNode(index);
        for (auto input : node->InputDefs()) {
          auto it = fused_outputs.find(input);
          if (it != fused_outputs.end()) {
            fused_outputs.erase(it);
          } else {
            fused_inputs.insert(input);
          }
        }
        for (auto output : node->OutputDefs()) {
          auto it = fused_inputs.find(output);
          if (it != fused_inputs.end()) {
            fused_inputs.erase(it);
          } else {
            fused_outputs.insert(output);
          }
        }
      }

      auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
      meta_def->name = "TVMFuseAdd";
      meta_def->domain = "FuseTest";
      for (auto input : fused_inputs) {
        meta_def->inputs.push_back(input->Name());
      }

      for (auto output : fused_outputs) {
        meta_def->outputs.push_back(output->Name());
      }

      meta_def->since_version = 1;
      meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
      sub_graph->SetMetaDef(meta_def);
      //TODO:set fuse kernel func;
      capacities.push_back(
          std::make_unique<ComputationCapacity>(std::move(sub_graph),
                                                [](const OpKernelInfo& info) -> OpKernel* { return new TVMFuseAddKernels<DefaultTVMScheduleGenerator, BuildStackVMDefaultModule>(info); }));
    }
  }
}

namespace test {

static void RunSession(InferenceSession& session_object,
                       RunOptions& run_options,
                       std::vector<int64_t>& dims_x,
                       std::vector<double>& values_x,
                       std::vector<int64_t>& dims_y,
                       std::vector<double>& values_y) {
  // prepare inputs
  MLValue ml_value;
  CreateMLValue<double>(TestCPUExecutionProvider()->GetAllocator(0, ONNXRuntimeMemTypeDefault), dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X1", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y4");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  EXPECT_EQ(expected_shape, rtensor.Shape());
  const std::vector<double> found(rtensor.template Data<double>(), rtensor.template Data<double>() + expected_shape.Size());
  ASSERT_EQ(found.size(), values_y.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_EQ(found[i], values_y[i]);
}

static const std::string MODEL_URI = "testdata/fuse_add_1.pb";

TEST(TVMTest, Fuse_Add_Test) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  CPUExecutionProviderInfo info;

  auto cpu_xp = std::make_unique<CPUExecutionProvider>(info);
  cpu_xp->InsertFusedRules(FuseAdd);
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(cpu_xp)).IsOK());
  EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
  EXPECT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {
      6,
  };
  std::vector<double> values_x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {
      6,
  };
  // now the expected value should be Add's result.
  std::vector<double> expected_values_y = {5.0, 10.0, 15.0, 20.0, 25.0, 30.0};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}
}  // namespace test

}  // namespace onnxruntime

TEST(TVMTest, Basic) {
  using namespace tvm;
  auto n = var("n");
  Array<Expr> shape;
  shape.push_back(n);
  auto A = placeholder(shape, Float(64), "A");
  auto B = placeholder(shape, Float(64), "B");
  auto D = placeholder(shape, Float(64), "D");
  auto C = compute(A->shape, [&A, &B](Expr i) {
    return A[i] + B[i];
  },
                   "C");
  auto E = compute(A->shape, [&C, &D](Expr i) {
    return C[i] + D[i];
  },
                   "E");

  auto s = create_schedule({E->op});
  auto args = Array<Tensor>({A, B, D, E});
  std::unordered_map<Tensor, Buffer> binds;
  auto config = build_config();
#ifdef USE_TVM_WITH_LLVM
  auto target = target::llvm();
#else
  auto target = target::stackvm();
#endif
  auto lowered = lower(s, args, "func", binds, config);
  auto module = build(lowered, target, Target(), config);
  auto func = module.GetFunction("func");

  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 64;
  dtype.lanes = 1;
  DLContext ctx;
  ctx.device_type = DLDeviceType::kDLCPU;
  ctx.device_id = 0;

  std::vector<double> v = {1.0, 2.0, 3.0};
  int64_t len = 3;
  DLTensor tensor_A = {&v[0], ctx, 1, dtype, &len, nullptr, 0};
  DLTensor tensor_B = {&v[0], ctx, 1, dtype, &len, nullptr, 0};
  DLTensor tensor_D = {&v[0], ctx, 1, dtype, &len, nullptr, 0};

  std::vector<double> r;
  r.resize(len);
  DLTensor tensor_E = {&r[0], ctx, 1, dtype, &len, nullptr, 0};

  TVMValue lvalues[4];
  int type_codes[4] = {kNDArrayContainer, kNDArrayContainer, kNDArrayContainer, kNDArrayContainer};
  lvalues[0].v_handle = &tensor_A;
  lvalues[1].v_handle = &tensor_B;
  lvalues[2].v_handle = &tensor_D;
  lvalues[3].v_handle = &tensor_E;

  TVMArgs tvm_args(lvalues, type_codes, 4);
  TVMRetValue rvalue;
  func.CallPacked(tvm_args, &rvalue);
  CHECK_EQ(rvalue.type_code(), kNull);
  double expected[3] = {3.0, 6.0, 9.0};
  auto data_E = static_cast<double*>(tensor_E.data);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(*(data_E + i), expected[i], 0.001f);
  }
}
