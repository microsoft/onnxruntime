// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/tvm/tvm_demo/demo_compiler.h"

#include <tvm/runtime/ndarray.h>

namespace onnxruntime {

using namespace tvm_demo;

class TVMDemoKernel : public OpKernel {
 public:
  explicit TVMDemoKernel(const OpKernelInfo& info) : OpKernel(info) {}

 protected:
  const TensorShape& GetOutputShape(OpKernelContext* context, int /*i*/) const {
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

static DLDataType GetDataType(ONNXTensorElementDataType type) {
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return {kDLFloat, 64, 1};
  } else
    ORT_THROW("not implement.");
}

namespace test {

struct TVMFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  tvm::runtime::Module* module = nullptr;
};

class FuseExecutionProviderX : public CPUExecutionProvider {
 public:
  explicit FuseExecutionProviderX(const CPUExecutionProviderInfo& info) : CPUExecutionProvider(info) {
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override {
    std::vector<std::unique_ptr<ComputeCapability>> result;
    std::vector<onnxruntime::NodeIndex> fused_nodes;
    for (auto& node : graph_viewer.Nodes()) {
      if (node.OpType() == "Mul") {
        fused_nodes.push_back(node.Index());
      }
    }

    UnionSet set(static_cast<int>(fused_nodes.size()));
    for (int i = 0; i < fused_nodes.size(); ++i) {
      auto node = graph_viewer.GetNode(fused_nodes[i]);
      for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
        auto index_it = std::find(fused_nodes.begin(), fused_nodes.end(), (*it).Index());
        if (index_it != fused_nodes.end()) {
          set.merge(i, static_cast<int>(index_it - fused_nodes.begin()));
        }
      }
    }

    std::vector<std::vector<onnxruntime::NodeIndex>> groups;
    groups.resize(fused_nodes.size());
    for (int i = 0; i < set.farthers_.size(); ++i) {
      groups[set.get(i)].push_back(fused_nodes[i]);
    }

    for (auto& group : groups) {
      if (group.size() > 1) {
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        std::set<const onnxruntime::NodeArg*> fused_inputs, fused_outputs;
        for (auto index : group) {
          sub_graph->nodes.push_back(index);
          auto node = graph_viewer.GetNode(index);
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
        meta_def->name = "TVMFuse";
        meta_def->domain = "FuseTest";
        for (auto input : fused_inputs) {
          meta_def->inputs.push_back(input->Name());
        }

        for (auto output : fused_outputs) {
          meta_def->outputs.push_back(output->Name());
        }

        meta_def->since_version = 1;
        meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
        sub_graph->SetMetaDef(std::move(meta_def));
        //TODO:set fuse kernel func;
        result.push_back(
            std::make_unique<ComputeCapability>(std::move(sub_graph)));
      }
    }
    return result;
  }

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override {
    for (auto* fused_node : fused_nodes) {
      auto func_body = fused_node->GetFunctionBody();
      if (!func_body)
        return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
      //1. Build tvm IR based on the Ort graph
      auto demo_tvm_tensor_ctx = BuildTVMIR(func_body->Body());
      //2. Create schedule for the built tvm IRs
      auto s = CreateSchedule(demo_tvm_tensor_ctx);
      //3. Build tvm module
      std::vector<tvm::Tensor> tvm_args;
      for (auto& t : demo_tvm_tensor_ctx.inputs) {
        tvm_args.push_back(t);
      }
      for (auto& t : demo_tvm_tensor_ctx.outputs) {
        tvm_args.push_back(t);
      }

      std::vector<std::string> func_names;
      auto module_ptr = std::make_shared<tvm::runtime::Module>();
      *module_ptr = BuildStackVMModule(s, tvm::build_config(), tvm_args, func_names);
      modules_[fused_node->Name()] = module_ptr;

      NodeComputeInfo compute_info;

      compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
        auto* p = new TVMFuncState();
        *p = {context->allocate_func, context->release_func, context->allocator_handle, modules_[context->node_name].get()};
        *state = p;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState state) {
        if (state)
          delete static_cast<TVMFuncState*>(state);
      };

      //we use lambda to capture the tvm model, so we can use it to get the funciton.
      compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
        Ort::CustomOpApi ort{*api};

        TVMFuncState* tvm_state = reinterpret_cast<TVMFuncState*>(state);

        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<int64_t>> output_shapes;

        auto eval_func_name = "func";
        DLContext cpu_context = {kDLCPU, 0};
        size_t num_inputs = ort.KernelContext_GetInputCount(context);
        size_t num_outputs = ort.KernelContext_GetOutputCount(context);
        size_t n_args = num_inputs + num_outputs;
        std::vector<DLTensor> dl_tensors(n_args);
        std::vector<TVMValue> tvm_values(n_args);
        std::vector<int> tvm_type_codes(n_args);
        for (auto i = 0; i < num_inputs; i++) {
          const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
          auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
          auto tensor_type = ort.GetTensorElementType(tensor_info);
          input_shapes.emplace_back(ort.GetTensorShape(tensor_info));
          ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

          tvm_type_codes[i] = kNDArrayContainer;
          dl_tensors[i].ctx = cpu_context;
          dl_tensors[i].dtype = GetDataType(tensor_type);
          dl_tensors[i].strides = nullptr;
          dl_tensors[i].byte_offset = 0;
          dl_tensors[i].data = const_cast<double*>(ort.GetTensorData<double>(input_tensor));
          dl_tensors[i].ndim = input_shapes.back().size();
          dl_tensors[i].shape = input_shapes.back().data();
          tvm_values[i].v_handle = &dl_tensors[i];
        }

        for (auto i = 0; i < num_outputs; i++) {
          //setup output tensor property
          //todo: type should be set by framework.
          output_shapes.push_back(input_shapes[i]);
          OrtValue* output_tensor = ort.KernelContext_GetOutput(context, i, output_shapes[i].data(), output_shapes[i].size());
          auto tensor_info = ort.GetTensorTypeAndShape(output_tensor);
          auto tensor_type = ort.GetTensorElementType(tensor_info);
          ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

          tvm_type_codes[num_inputs + i] = kNDArrayContainer;
          dl_tensors[num_inputs + i].ctx = cpu_context;
          dl_tensors[num_inputs + i].dtype = GetDataType(tensor_type);
          dl_tensors[num_inputs + i].strides = nullptr;
          dl_tensors[num_inputs + i].byte_offset = 0;
          dl_tensors[num_inputs + i].data = ort.GetTensorMutableData<double>(output_tensor);
          dl_tensors[num_inputs + i].ndim = output_shapes.back().size();
          dl_tensors[num_inputs + i].shape = output_shapes.back().data();
          tvm_values[num_inputs + i].v_handle = &dl_tensors[num_inputs + i];
        }

        auto evaluate_func_ = tvm_state->module->GetFunction(eval_func_name);
        tvm::TVMArgs tvm_args(&tvm_values[0], &tvm_type_codes[0], static_cast<int>(n_args));
        tvm::TVMRetValue rvalue;
        try {
          evaluate_func_.CallPacked(tvm_args, &rvalue);
        } catch (std::exception&) {
          return Status(common::ONNXRUNTIME, common::FAIL);  // TODO: Translate exception to error code
        }
        if (rvalue.type_code() != kNull) {
          return Status(common::ONNXRUNTIME, common::FAIL);  // TODO: get error code.
        } else {
          return Status::OK();
        }
      };
      node_compute_funcs.push_back(compute_info);
    }

    return Status::OK();
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<tvm::runtime::Module>> modules_;
};

static void RunSession(InferenceSession& session_object,
                       RunOptions& run_options,
                       std::vector<int64_t>& dims_x,
                       std::vector<double>& values_x,
                       std::vector<int64_t>& dims_y,
                       std::vector<double>& values_y) {
  // prepare inputs
  OrtValue ml_value;
  CreateMLValue<double>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X1", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y4");
  std::vector<OrtValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
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

static const std::string MODEL_URI = "testdata/fuse_mul_1.onnx";

TEST(TVMTest, CodeGen_Demo_for_Fuse_Mul) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, GetEnvironment()};
  CPUExecutionProviderInfo info;
  auto tvm_xp = std::make_unique<FuseExecutionProviderX>(info);
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(tvm_xp)).IsOK());
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
  // now the expected value should be Mul's result.
  std::vector<double> expected_values_y = {1.0, 32.0, 243.0, 1024.0, 3125.0, 7776.0};

  // Now run
  RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}
}  // namespace test

}  // namespace onnxruntime

TEST(TVMTest, Native_TVM) {
  using namespace tvm;
  auto n = var("n");
  Array<Expr> shape;
  shape.push_back(n);
  auto A = placeholder(shape, Float(64), "A");
  auto B = placeholder(shape, Float(64), "B");
  auto D = placeholder(shape, Float(64), "D");
  auto C = compute(
      A->shape, [&A, &B](Expr i) {
        return A[i] + B[i];
      },
      "C");
  auto E = compute(
      A->shape, [&C, &D](Expr i) {
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
