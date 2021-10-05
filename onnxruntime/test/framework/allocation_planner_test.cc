// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include "gtest/gtest.h"

#include "core/framework/session_state.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "test/framework/model_builder_utils.h"
#include "core/framework/allocation_planner.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/util/thread_utils.h"

#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

using namespace ONNX_NAMESPACE;

// Explicitly provide a definition for the static const var 'GPU' in the OrtDevice struct,
// GCC 4.x doesn't seem to define this and it breaks the pipelines based on CentOS as it uses
// GCC 4.x.
// (This static var is referenced in some tests below)
const OrtDevice::DeviceType OrtDevice::GPU;

namespace onnxruntime {
namespace test {

namespace modelbuilder {

class NodeCounter {
 private:
  static int node_count_;

 public:
  static int Next() { return ++node_count_; }
};

int NodeCounter::node_count_ = 0;

struct UnaryNode {
  std::vector<onnxruntime::NodeArg*> input_args;
  std::vector<onnxruntime::NodeArg*> output_args;
  onnxruntime::Node* p_node;

  UnaryNode(onnxruntime::Graph& graph, const std::string& op, onnxruntime::NodeArg* p_input_arg,
            onnxruntime::NodeArg* p_output_arg)
      : input_args({p_input_arg}), output_args({p_output_arg}) {
    int num = NodeCounter::Next();
    p_node = &graph.AddNode("node" + std::to_string(num), op, "test op", input_args, output_args);
  }

  UnaryNode(onnxruntime::Graph& graph, onnxruntime::NodeArg* p_input_arg, onnxruntime::NodeArg* p_output_arg)
      : UnaryNode(graph, "Transpose", p_input_arg, p_output_arg) {}
};

class DummyOpKernel : public OpKernel {
 public:
  DummyOpKernel(const OpKernelInfo& p) : OpKernel(p) {}
  Status Compute(OpKernelContext* context) const {
    ORT_UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const {
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(done);
    return Status::OK();
  }
};

}  // namespace modelbuilder

using namespace modelbuilder;

class AllocationPlanTestUtility {
 public:
  static void CheckAllocationKind(const SequentialExecutionPlan& plan, std::vector<AllocKind>& expected) {
    ASSERT_EQ(plan.allocation_plan.size(), expected.size()) << "Allocation plan of wrong size";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(plan.allocation_plan[i].alloc_kind, expected[i]) << "Error in allocation kind at position " << i;
    }
  }

  static void CheckToBeFreed(const SequentialExecutionPlan& plan, const std::vector<OrtValueIndex>& expected) {
    ASSERT_EQ(plan.to_be_freed.size(), expected.size()) << "Allocation plan's to_be_freed of wrong size";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(plan.to_be_freed[i], expected[i]) << "Error in to_be_freed at position " << i;
    }
  }

  static void CheckFreedAtEachStep(const SequentialExecutionPlan& plan, const std::vector<int>& expected_num_freed) {
    ASSERT_EQ(plan.execution_plan.size(), expected_num_freed.size()) << "Execution plan is of wrong size";
    int start = 0;
    for (size_t i = 0; i < expected_num_freed.size(); ++i) {
      if (expected_num_freed[i] > 0) {
        EXPECT_EQ(plan.execution_plan[i].free_from_index, start) << "Error in free_from_index at position " << i;
        EXPECT_EQ(plan.execution_plan[i].free_to_index, start + expected_num_freed[i] - 1)
            << "Error in free_to_index at position " << i;
        start = start + expected_num_freed[i];
      } else {
        // "free_from_index > free_to_index" indicates nothing is to be freed
        EXPECT_GT(plan.execution_plan[i].free_from_index, plan.execution_plan[i].free_to_index);
      }
    }
  }

  static void BasicIntegrityCheck(const SequentialExecutionPlan& plan, size_t num_ml_values) {
    // Sanity checks for plan.to_be_freed
    std::unordered_set<OrtValueIndex> freed;
    for (OrtValueIndex index : plan.to_be_freed) {
      // Every index should be in the valid range [0, num_ml_values-1]
      EXPECT_GE(index, 0);
      EXPECT_LT(static_cast<size_t>(index), num_ml_values);
      // An index should not be freed more than once
      EXPECT_EQ(freed.count(index), 0u) << "OrtValue " << index << " freed multiple times";
      freed.insert(index);
    }
    // Check the free-index information for every execution step: they should cover the
    // range [0, plan.to_be_freed.size()-1] properly.
    int next_free_index = 0;
    int max_free_index = ((int)plan.to_be_freed.size()) - 1;
    for (const SequentialExecutionPlan::NodeExecutionPlan& step : plan.execution_plan) {
      if (step.free_from_index <= step.free_to_index) {
        EXPECT_EQ(step.free_from_index, next_free_index);
        EXPECT_LE(step.free_to_index, max_free_index);
        next_free_index = step.free_to_index + 1;
      }  // else nothing needs to be freed in this step
    }
  }
};

typedef std::unordered_map<const onnxruntime::NodeArg*, TensorShapeProto*> ShapeMap;

class SequentialPlannerTestContext : public ISequentialPlannerContext {
 public:
  SequentialPlannerTestContext(ShapeMap* shape_map) : shape_map_(shape_map) {}

  TensorShapeProto* GetShape(const onnxruntime::NodeArg& arg) const override {
    auto iter = shape_map_->find(&arg);
    return (shape_map_->end() != iter) ? iter->second : nullptr;
  }

 private:
  ShapeMap* shape_map_;
};

class PlannerTest : public ::testing::Test {
 private:
  void index(const std::string& name, int& out) {
    ASSERT_TRUE(state_->GetOrtValueNameIdxMap().GetIdx(name, out).IsOK());
  }

  onnxruntime::Model model_;
  onnxruntime::Graph& graph_;

  // some standard components used to build test-cases:
  Type float_type_;

  std::unique_ptr<::onnxruntime::KernelDef> std_kernel_;               // a unary kernel with no-aliasing and no-in-place
  std::unique_ptr<::onnxruntime::KernelDef> in_place_kernel_;          // a unary kernel with in-place
  std::unique_ptr<::onnxruntime::KernelDef> external_outputs_kernel_;  // an unary kernel with external outputs

  std::unordered_map<std::string, onnxruntime::NodeArg*> name_to_arg_;
  std::vector<std::unique_ptr<UnaryNode>> nodes_;
  std::vector<std::unique_ptr<OpKernelInfo>> op_kernel_infos_;
  std::vector<std::pair<onnxruntime::Node*, KernelDef&>> kernel_bindings_;
  ExecutionProviders execution_providers_;
  std::unique_ptr<concurrency::ThreadPool> tp_;
  DataTransferManager dtm_;
  profiling::Profiler profiler_;
  std::unique_ptr<SessionState> state_;
  ShapeMap shape_map_;
  std::unique_ptr<SequentialExecutionPlan> plan_;

 public:
  PlannerTest()
      : model_("test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 10}}, {}, DefaultLoggingManager().DefaultLogger()),
        graph_(model_.MainGraph()),
        tp_(concurrency::CreateThreadPool(&onnxruntime::Env::Default(), OrtThreadPoolParams(),
                                          concurrency::ThreadPoolType::INTRA_OP)) {
    std_kernel_ = KernelDefBuilder().SetName("Transpose").Provider(kCpuExecutionProvider).SinceVersion(1, 10).Build();
    in_place_kernel_ =
        KernelDefBuilder().SetName("Relu").Provider(kCpuExecutionProvider).SinceVersion(1, 10).MayInplace(0, 0).Build();
    external_outputs_kernel_ =
        KernelDefBuilder().SetName("Tanh").Provider(kCpuExecutionProvider).SinceVersion(1, 10).ExternalOutputs().Build();
    CPUExecutionProviderInfo epi;
    auto execution_provider = std::make_unique<CPUExecutionProvider>(epi);
    ORT_THROW_IF_ERROR(execution_providers_.Add("CPUExecutionProvider", std::move(execution_provider)));

    state_.reset(new SessionState(graph_, execution_providers_, false, tp_.get(), nullptr, dtm_,
                                  DefaultLoggingManager().DefaultLogger(), profiler_));
  }

  onnxruntime::NodeArg* Arg(const std::string& name) {
    auto iter = name_to_arg_.find(name);
    if (name_to_arg_.end() != iter) return iter->second;
    return (name_to_arg_[name] = &graph_.GetOrCreateNodeArg(name, &float_type_.value));
  }

  onnxruntime::Node* AddNode(::onnxruntime::KernelDef& kernel_def, std::string& input, std::string& output) {
    auto node = std::make_unique<UnaryNode>(graph_, kernel_def.OpName(), Arg(input), Arg(output));
    auto* p_node = node->p_node;
    p_node->SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
    nodes_.push_back(std::move(node));
    kernel_bindings_.emplace_back(p_node, kernel_def);
    return p_node;
  }

  onnxruntime::Node* AddNormalNode(std::string& input, std::string& output) {
    return AddNode(*std_kernel_, input, output);
  }

  onnxruntime::Node* AddInplaceNode(std::string& input, std::string& output) {
    return AddNode(*in_place_kernel_, input, output);
  }

  onnxruntime::Node* AddExternalOutputsNode(std::string& input, std::string& output) {
    return AddNode(*external_outputs_kernel_, input, output);
  }

  void BindKernel(onnxruntime::Node* p_node, ::onnxruntime::KernelDef& kernel_def, KernelRegistry* reg,
                  std::unordered_map<NodeIndex, gsl::not_null<const KernelCreateInfo*>>& kernel_create_info_map) {
    const IExecutionProvider* ep = execution_providers_.Get(*p_node);
    ASSERT_NE(ep, nullptr);
    auto info = std::make_unique<OpKernelInfo>(
        *p_node, kernel_def, *ep, state_->GetInitializedTensors(), state_->GetOrtValueNameIdxMap(),
        state_->GetFuncMgr(), state_->GetDataTransferMgr());

    op_kernel_infos_.push_back(std::move(info));
    if (!KernelRegistry::HasImplementationOf(*reg, *p_node, onnxruntime::kCpuExecutionProvider)) {
      auto st = reg->Register(
          KernelCreateInfo(std::make_unique<KernelDef>(kernel_def),
                           [](const OpKernelInfo& info) -> OpKernel* { return new DummyOpKernel(info); }));
      ORT_ENFORCE(st.IsOK(), st.ErrorMessage());
    }

    const KernelCreateInfo* kci;
    ASSERT_STATUS_OK(reg->TryFindKernel(*p_node, "", &kci));
    kernel_create_info_map.insert({p_node->Index(), gsl::not_null<const KernelCreateInfo*>(kci)});
  }

  void SetShape(std::string& name, TensorShapeProto* shape) { shape_map_[Arg(name)] = shape; }

  void SetShape(std::initializer_list<std::pair<std::string&, TensorShapeProto*>> shapes) {
    for (auto& pair : shapes) {
      SetShape(pair.first, pair.second);
    }
  }

  void CreatePlan(const std::vector<const NodeArg*>& outer_scope_node_args = {}) {
    EXPECT_EQ(graph_.Resolve(), Status::OK());

    std::shared_ptr<KernelRegistry> reg = std::make_shared<KernelRegistry>();
    std::unordered_map<NodeIndex, gsl::not_null<const KernelCreateInfo*>> kernel_create_info_map;

    for (auto& binding : kernel_bindings_) {
      BindKernel(binding.first, binding.second, reg.get(), kernel_create_info_map);
    }

    auto cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    KernelRegistryManager kernel_registry_manager;
    kernel_registry_manager.RegisterKernelRegistry(reg);
    auto status = kernel_registry_manager.RegisterKernels(execution_providers_);
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

    // CreatePlan is called inside FinalizeSessionState and usually the initializers are removed following that.
    // Leave initializers so we can duplicate the call to CreatePlan from here to validate.
    const bool remove_initializers = false;
    status = state_->FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager, {}, nullptr, remove_initializers);

    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    SequentialPlannerTestContext test_context(&shape_map_);

    status = SequentialPlanner::CreatePlan(nullptr, GraphViewer(graph_), outer_scope_node_args, execution_providers_,
                                           kernel_create_info_map, {}, {}, state_->GetOrtValueNameIdxMap(), test_context,
                                           plan_);

    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    AllocationPlanTestUtility::BasicIntegrityCheck(*plan_, name_to_arg_.size());
  }

  void CheckAllocKind(const std::string& name, AllocKind kind) {
    int id;
    index(name, id);
    EXPECT_EQ(plan_->allocation_plan[id].alloc_kind, kind) << "Error in allocation kind for " << name;
  }

  void CheckFreed(int step_number, std::initializer_list<std::string> freed_items) {
    // create set and check equality
    std::unordered_set<int> expected;
    for (auto& name : freed_items) {
      int id;
      index(name, id);
      expected.insert(id);
    }
    std::unordered_set<int> plan_result;
    auto& step_plan = plan_->execution_plan[step_number];
    for (int i = step_plan.free_from_index; i <= step_plan.free_to_index; ++i)
      plan_result.insert(plan_->to_be_freed[i]);
    EXPECT_EQ(plan_result, expected) << "Freed items incorrect for step " << step_number;
  }

 protected:
  Graph& GetGraph() { return graph_; }
  const SequentialExecutionPlan& GetPlan() const { return *plan_; }
  const SessionState& GetState() const { return *state_; }
};

TEST_F(PlannerTest, ChainTest) {
  // tensor variables:
  std::string W("W"), X("X"), B("B"), Y("Y"), Z("Z");

  // graph structure:

  ONNX_NAMESPACE::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("W");
  GetGraph().AddInitializedTensor(tensor);

  AddNormalNode(W, X);
  AddNormalNode(X, B);
  AddNormalNode(B, Y);
  AddNormalNode(Y, Z);

  // simulate shape-inference results:
  Shape shape1{50, 100};
  auto shape = &shape1.value;
  SetShape({{X, shape}, {B, shape}, {Y, shape}, {Z, shape}});

  CreatePlan();

  // Expected plan:
  //   W: kAllocateStatically; X: kAllocate; B: kAllocate; Y: kReuse (X); post-node3: free(B); X is returned output
  CheckAllocKind(W, AllocKind::kAllocateStatically);
  CheckAllocKind(X, AllocKind::kAllocate);
  CheckAllocKind(B, AllocKind::kAllocate);
  CheckAllocKind(Y, AllocKind::kReuse);
  CheckAllocKind(Z, AllocKind::kAllocateOutput);

  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {"B"});
  CheckFreed(3, {"X"});
}

/* InputOutputTest: Test that:
(a) All inputs are classified as kPreExisting,
(b) All outer scope node args are classified as kPreExisting,
(c) All outputs are classified as kAllocate (in this example),
(d) Neither input nor outputs are freed.
*/
TEST_F(PlannerTest, InputOutputTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), Y1("Y1"), Y2("Y2"), Outer1("Outer1"), Y3("Y3");

  // graph structure:
  AddNormalNode(X1, Y1);
  AddNormalNode(X2, Y2);

  // add node that consumes an outer scope node arg
  auto outer_node = AddNormalNode(Outer1, Y3);
  const NodeArg* outer_scope_node_arg = outer_node->InputDefs().at(0);
  GetGraph().AddOuterScopeNodeArg(Outer1);

  // simulate no shape-inference:

  CreatePlan({outer_scope_node_arg});

  // X1: kPreExisting, X2: kPreExisting, Outer1: kPreExisting, Y1: kAllocate, Y2: kAllocate, Y3: kAllocate
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kPreExisting);
  CheckAllocKind(Outer1, AllocKind::kPreExisting);
  CheckAllocKind(Y1, AllocKind::kAllocateOutput);
  CheckAllocKind(Y2, AllocKind::kAllocateOutput);
  CheckAllocKind(Y3, AllocKind::kAllocateOutput);

  // Nothing should be freed (since they are either inputs or outputs)
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {});
}

// InPlaceTest: Check that we reuse when Inplace allows us to.

TEST_F(PlannerTest, InPlaceTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kReuse);
  CheckAllocKind(X4, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {X2});
}

TEST_F(PlannerTest, ExternalOutputsTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddExternalOutputsNode(X1, X2);  // external-outputs operator; X1: input; X2: temporary
  AddNormalNode(X2, X3);           // normal operator; X3: temporary
  AddNormalNode(X3, X4);           // normal operator; X4: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocatedExternally);
  CheckAllocKind(X3, AllocKind::kAllocate);
  CheckAllocKind(X4, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  // X2 will not be reused and will not be freed. X3 will be allocated and will be freed.
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {X3});
}

// InPlaceSizeMismatchTest: Check that Inplace reuse is not allowed when sizes don't match.
// Also tests reuse of disjoint lifetime tensors.
TEST_F(PlannerTest, InPlaceSizeMismatchTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4"), X5("X5");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: temporary
  AddInplaceNode(X4, X5);  // may-in-place operator; X5 output

  // simulate shape-inference results:
  Shape shape1w{"M", "N"};
  auto shape1 = &shape1w.value;
  Shape shape2w{"M", "K"};
  auto shape2 = &shape2w.value;
  SetShape({{X1, shape1}, {X2, shape1}, {X3, shape2}, {X4, shape1}, {X5, shape1}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kAllocate);
  CheckAllocKind(X4, AllocKind::kReuse);
  CheckAllocKind(X5, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {X3});
  CheckFreed(3, {X2});
}

// Test operator<< to output details of an allocation & execution plan.
TEST_F(PlannerTest, PlanOutputTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  ORT_TRY {
    std::ostringstream output;
    output << std::make_pair(&GetPlan(), &GetState());
    auto output_size = output.str().size();
    // Currently, we don't check details of the output, as it may change over time.
    EXPECT_GT(output_size, 0u);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&ex]() {
      EXPECT_TRUE(false) << "Exception in producing output: " << ex.what();
    });
  }
}

#ifdef USE_CUDA
TEST_F(PlannerTest, LocationPlanningForPassThroughExplicitAndImplicitSubgraphInputs) {
  // Types
  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim_param");

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // The model has a main graph and 2 levels of nested subgraphs
  // Main graph: 2 Abs nodes + one Loop node
  // First level (Loop) subgraph: Identity (condition pass-through) + If node
  // Second level subgraph(s): Then and Else branches: Both have an Add node
  // The Add node adds 2 values:
  // One value from the main graph ("abs_data_0_out") that is "implicitly"
  // consumed by the Loop node and "passed through" to the If subgraphs.
  // Another value from the main graph ("abs_data_1_out") that is "explicitly"
  // consumed by the Loop node as a loop carried dependency and its name in
  // the scope of the Loop node is "loop_state_var".

  // In the Loop subgraph, there are no explicit consumers of "abs_data_0_out"
  // and "loop_state_var", there is only one implicit consumer - "If".
  // We want to ensure that since there are no explicit consumers, the planned locations
  // for these values in this subgraph are the same locations as their corresponding
  // values in the outer scope, thus deferring any copies (if required) till the actual
  // subgraph(s) they are explicitly consumed in.
  auto create_model = [&float_tensor, &int64_scalar, &bool_scalar]() -> Model {
    auto create_if_subgraph = [&float_tensor](bool is_then) -> GraphProto {
      Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      auto& outer_scope_0 = graph.GetOrCreateNodeArg("loop_state_var", &float_tensor);
      graph.AddOuterScopeNodeArg("loop_state_var");

      auto& outer_scope_1 = graph.GetOrCreateNodeArg("abs_data_0_out", &float_tensor);
      graph.AddOuterScopeNodeArg("abs_data_0_out");

      auto& if_out = graph.GetOrCreateNodeArg(is_then ? "if_then_out" : "if_else_out", &float_tensor);
      graph.AddNode("if_out", "Add", "add", {&outer_scope_0, &outer_scope_1}, {&if_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    auto create_loop_subgraph = [&create_if_subgraph, &float_tensor, &int64_scalar, &bool_scalar]() -> GraphProto {
      Model model("loop_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      std::vector<NodeArg*> inputs;
      std::vector<NodeArg*> outputs;

      /*  Inputs: iter_num, cond_in, loop carried state variables.
         iter_num_in    cond_in     [loop_state_var]
           (unused)        |               |
                       [Identity]         [If]  
                           |               |
                        cond_out     loop_state_var_out
    */

      // graph inputs
      auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
      auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
      auto& loop_state_var = graph.GetOrCreateNodeArg("loop_state_var", &float_tensor);

      // graph outputs
      auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
      auto& loop_state_var_out = graph.GetOrCreateNodeArg("loop_state_var_out", &float_tensor);

      // outer scope args
      ORT_IGNORE_RETURN_VALUE(graph.GetOrCreateNodeArg("abs_data_0_out", &float_tensor));
      graph.AddOuterScopeNodeArg("abs_data_0_out");

      // cond_in -> cond_out
      {
        inputs = {&cond_in};
        outputs = {&cond_out};

        graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
      }

      // loop_state_var -> If(cond_in) -> loop_state_var_out
      {
        inputs = {&cond_in};
        outputs = {&loop_state_var_out};

        auto& node = graph.AddNode("loop_var_out", "If", "If with loop_state_var as implicit_input", inputs, outputs);
        node.AddAttribute("then_branch", create_if_subgraph(true));
        node.AddAttribute("else_branch", create_if_subgraph(false));
      }

      graph.SetInputs({&iter_num_in, &cond_in, &loop_state_var});
      graph.SetOutputs({&cond_out, &loop_state_var_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    onnxruntime::Model model("main_graph", false, ModelMetaData(),
                             PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                             {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& main_graph = model.MainGraph();

    // Abs-0
    auto& abs_data_0_in = main_graph.GetOrCreateNodeArg("abs_data_0_in", &float_tensor);
    auto& abs_data_0_out = main_graph.GetOrCreateNodeArg("abs_data_0_out", &float_tensor);
    std::vector<onnxruntime::NodeArg*> abs_0_inputs = {&abs_data_0_in};
    std::vector<onnxruntime::NodeArg*> abs_0_outputs = {&abs_data_0_out};
    main_graph.AddNode("abs_0", "Abs", "node abs", abs_0_inputs, abs_0_outputs);

    // Abs-1
    auto& abs_data_1_in = main_graph.GetOrCreateNodeArg("abs_data_1_in", &float_tensor);
    auto& abs_data_1_out = main_graph.GetOrCreateNodeArg("abs_data_1_out", &float_tensor);
    std::vector<onnxruntime::NodeArg*> abs_1_inputs = {&abs_data_1_in};
    std::vector<onnxruntime::NodeArg*> abs_1_outputs = {&abs_data_1_out};
    main_graph.AddNode("abs_1", "Abs", "node abs", abs_1_inputs, abs_1_outputs);

    // Loop
    auto& iter_num_in = main_graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = main_graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
    auto& loop_state_out_var = main_graph.GetOrCreateNodeArg("loop_state_out_var", &float_tensor);

    auto& loop_node = main_graph.AddNode("loop", "Loop", "Loop node",
                                         {&iter_num_in, &cond_in, &abs_data_1_out},
                                         {&loop_state_out_var});
    loop_node.AddAttribute("body", create_loop_subgraph());

    main_graph.SetInputs({&abs_data_0_in, &abs_data_1_in, &iter_num_in, &cond_in});
    main_graph.SetOutputs({&loop_state_out_var});

    auto status = main_graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return model;
  };

  // Create and load session
  SessionOptions so;
  InferenceSession sess{so, GetEnvironment()};

  auto status = sess.RegisterExecutionProvider(DefaultCudaExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  std::string s1;
  const bool rc = create_model().ToProto().SerializeToString(&s1);
  EXPECT_EQ(rc, true);
  std::stringstream sstr(s1);

  status = sess.Load(sstr);
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Check planned locations of values in the main graph that are implicit subgraph inputs
  // and explicit subgraph inputs to the Loop node

  // Main graph (L0 graph)
  const auto& main_graph_session_state = sess.GetSessionState();

  {
    const auto& main_graph_ort_value_index_map = main_graph_session_state.GetOrtValueNameIdxMap();
    const auto* main_graph_plan = main_graph_session_state.GetExecutionPlan();

    OrtValueIndex abs_data_0_out_index;
    ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("abs_data_0_out", abs_data_0_out_index));

    OrtValueIndex abs_data_1_out_index;
    ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("abs_data_1_out", abs_data_1_out_index));

    EXPECT_EQ(main_graph_plan->allocation_plan[abs_data_0_out_index].location.device.Type(), OrtDevice::GPU);
    EXPECT_EQ(main_graph_plan->allocation_plan[abs_data_1_out_index].location.device.Type(), OrtDevice::GPU);
  }

  // First subgraph (Loop) (L1 graph)
  // There are 3 nodes in the main level- Only one of them has a subgraph (Loop).
  // Find that.
  const SessionState* find_first_subgraph_session_state = nullptr;
  for (size_t i = 0; i < 3; ++i) {
    find_first_subgraph_session_state = main_graph_session_state.GetSubgraphSessionState(i, "body");
    if (find_first_subgraph_session_state) {
      break;
    }
  }

  const auto& first_subgraph_session_state = *find_first_subgraph_session_state;

  {
    const auto& first_subgraph_ort_value_index_map = first_subgraph_session_state.GetOrtValueNameIdxMap();
    const auto* first_subgraph_plan = first_subgraph_session_state.GetExecutionPlan();

    OrtValueIndex abs_data_0_out_index;
    ASSERT_STATUS_OK(first_subgraph_ort_value_index_map.GetIdx("abs_data_0_out", abs_data_0_out_index));

    // "abs_data_1_out" is "loop_state_var" in this scope as it was consumed as an explicit subgraph input
    // to Loop's body subgraph
    OrtValueIndex abs_data_1_out_index;
    ASSERT_STATUS_OK(first_subgraph_ort_value_index_map.GetIdx("loop_state_var", abs_data_1_out_index));

    // There are no explicit consumers of "abs_data_0_out" and "loop_state_var (abs_data_1_out)" in this scope.
    // There is only one implicit consumer "If". Hence, check that we are preserving the locations of these values
    // from the outer scope, thus deferring any copies till the actual nested subgraph these values are used in.
    EXPECT_EQ(first_subgraph_plan->allocation_plan[abs_data_0_out_index].location.device.Type(), OrtDevice::GPU);
    EXPECT_EQ(first_subgraph_plan->allocation_plan[abs_data_1_out_index].location.device.Type(), OrtDevice::GPU);
  }
}
TEST_F(PlannerTest, LocationPlanningForInitializersOnlyUsedInANestedSubgraph) {
  // This a simple model that has one outer scope initializer and an `If` node
  // and that initializer is ONLY used in nested subgraphs (both the `If` subgraphs).
  // We want to test that the location planned for this initializer accounts for
  // its usage in the nested subgraphs and statically determines the right location
  // for it (without defaulting to CPU).

  // Types
  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim_param");

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_model = [&float_tensor, &bool_scalar]() -> Model {
    auto create_if_subgraph = [&float_tensor](bool is_then) -> GraphProto {
      Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      auto& outer_scope_0 = graph.GetOrCreateNodeArg("abs_data_out", &float_tensor);
      graph.AddOuterScopeNodeArg("abs_data_out");

      auto& outer_scope_1 = graph.GetOrCreateNodeArg("init_data", &float_tensor);
      graph.AddOuterScopeNodeArg("init_data");

      auto& if_out = graph.GetOrCreateNodeArg(is_then ? "if_then_out" : "if_else_out", &float_tensor);
      graph.AddNode("if_out", "Add", "add", {&outer_scope_0, &outer_scope_1}, {&if_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    onnxruntime::Model model("main_graph", false, ModelMetaData(),
                             PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                             {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& main_graph = model.MainGraph();

    // Abs-0
    auto& abs_data_in = main_graph.GetOrCreateNodeArg("abs_data_in", &float_tensor);
    auto& abs_data_out = main_graph.GetOrCreateNodeArg("abs_data_out", &float_tensor);
    main_graph.AddNode("abs_0", "Abs", "node abs", {&abs_data_in}, {&abs_data_out});

    // If
    auto& if_in = main_graph.GetOrCreateNodeArg("if_in", &bool_scalar);
    auto& if_out = main_graph.GetOrCreateNodeArg("if_out", &float_tensor);
    auto& node = main_graph.AddNode("if_out", "If", "If", {&if_in}, {&if_out});
    node.AddAttribute("then_branch", create_if_subgraph(true));
    node.AddAttribute("else_branch", create_if_subgraph(false));

    // Add initializer to the graph
    ONNX_NAMESPACE::TensorProto tensor;
    tensor.add_dims(1);
    tensor.add_float_data(1.0f);
    tensor.set_data_type(TensorProto_DataType_FLOAT);
    tensor.set_name("init_data");
    main_graph.AddInitializedTensor(tensor);

    // Main graph's inputs/outputs
    main_graph.SetInputs({&abs_data_in, &if_in});
    main_graph.SetOutputs({&if_out});

    auto status = main_graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return model;
  };

  // Create and load session
  SessionOptions so;
  InferenceSession sess{so, GetEnvironment()};

  auto status = sess.RegisterExecutionProvider(DefaultCudaExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  std::string s1;
  const bool rc = create_model().ToProto().SerializeToString(&s1);
  EXPECT_EQ(rc, true);
  std::stringstream sstr(s1);

  status = sess.Load(sstr);
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Check planned locations for the initializer
  const auto& main_graph_session_state = sess.GetSessionState();
  const auto& main_graph_ort_value_index_map = main_graph_session_state.GetOrtValueNameIdxMap();
  const auto* main_graph_plan = main_graph_session_state.GetExecutionPlan();

  OrtValueIndex init_data_index;
  ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("init_data", init_data_index));

  EXPECT_EQ(main_graph_plan->allocation_plan[init_data_index].location.device.Type(), OrtDevice::GPU);
}
#endif
}  // namespace test
}  // namespace onnxruntime
