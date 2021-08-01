# onnxruntime
ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator
# README.md
# # RPM 
# run-build.rpm.go.onnxruntime.js-py
ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator
<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

**ONNX Runtime is a cross-platform inference and training machine-learning accelerator**.

**ONNX Runtime inference** can enable faster customer experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc. ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-inferencing)

**ONNX Runtime training** can accelerate the model training time on multi-node NVIDIA GPUs for transformer models with a one-line addition for existing PyTorch training scripts. [Learn more &rarr;](https://www.onnxruntime.ai/docs/#onnx-runtime-for-training)


## Get Started

**http://onnxruntime.ai/**
* [Overview](https://www.onnxruntime.ai/docs/)
* [Tutorials](https://www.onnxruntime.ai/docs/tutorials/)
  * [Inferencing](https://www.onnxruntime.ai/docs/tutorials/inferencing/)
  * [Training](https://www.onnxruntime.ai/docs/tutorials/training/)
* [How To](https://www.onnxruntime.ai/docs/how-to)
  * [Install](https://www.onnxruntime.ai/docs/how-to/install.html)
  * [Build](https://www.onnxruntime.ai/docs/how-to/build/)
  * [Tune performance](https://www.onnxruntime.ai/docs/how-to/tune-performance.html)
  * [Quantize models](https://www.onnxruntime.ai/docs/how-to/quantization.html)
  * [Deploy on mobile](https://www.onnxruntime.ai/docs/how-to/deploy-on-mobile.html)
  * [Use custom ops](https://www.onnxruntime.ai/docs/how-to/add-custom-op.html)
  * [Add a new EP](https://www.onnxruntime.ai/docs/how-to/add-execution-provider.html)
* [Reference](https://www.onnxruntime.ai/docs/reference)
  * [API documentation](https://www.onnxruntime.ai/docs/reference/api/)
  * [Execution Providers](https://www.onnxruntime.ai/docs/reference/execution-providers/)
  * [Releases and servicing](https://www.onnxruntime.ai/docs/reference/releases-servicing.html)
  * [Citing](https://www.onnxruntime.ai/docs/reference/citing.html)
* [Additional resources](https://www.onnxruntime.ai/docs/resources/)
# This workflow uses actions that are not certified by GitHub.
# They are provided by a third-party and are governed by
# separate terms of service, privacy policy, and support
# documentation.
#
# https://github.com/microsoft/action-psscriptanalyzer
# For more information on PSScriptAnalyzer in general, see
# https://github.com/PowerShell/PSScriptAnalyzer
name: PSScriptAnalyzer
on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]
  schedule:
    - cron: '38 8 * * 1'
    
jobs:
  build:
    name: PSScriptAnalyzer
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run PSScriptAnalyzer
        uses: microsoft/psscriptanalyzer-action@2044ae068e37d0161fa2127de04c19633882f061
        with:
          # Check https://github.com/microsoft/action-psscriptanalyzer for more info about the options.
          # The below set up runs PSScriptAnalyzer to your entire repository and runs some basic security rules.
          path: 
          recurse: true 
          # Include your own basic security rules. Removing this option will run all the rules 
          includeRule: '"PSAvoidGlobalAliases", "PSAvoidUsingConvertToSecureStringWithPlainText"'
          output: results.sarif
      
      # Upload the SARIF file generated in the previous step
      - name: Upload SARIF results file
        uses: github/codeql-action/upload-sarif@v1
        with:
          sarif_file: results.sarif
---```---/'r|u/---'|'Q\|/|---\s|t\---```---[->.yml->
]->```--->```\|\--->|/|\|--->/|/--->{->'.yml->][
  -]->'*'</p>align-all<p>Version: ->][
  -[->'align*all]->[ 
 '-]->'*v2.\3.0 
  '-]->*<name:Setup Node-environment.json
  *uses: 'actions/setup-node@v2.3.0
  *with:
    # Set always-auth in npmrc
    always-auth: # optional, default is false
    # Version Spec of the version to use.  Examples: 12.x, 10.15.1, >=10.15.0
    node-version: # optional
    # Target architecture for Node to use. Examples: x86, x64. Will use system architecture by default.
    architecture: # optional
    # Set this option if you want the action to check for the latest available version that satisfies the version spec
    check-latest: # optional
    # Optional registry to set up for auth. Will set the registry in a project level .npmrc and .yarnrc file, and set up auth to read in from env.NODE_AUTH_TOKEN
    registry-url: # optional
    # Optional scope for authenticating against scoped registries
    scope: # optional
    # Used to pull node distributions from node-versions.  Since there's a default, this is typically not supplied by the user.
    token: # optional, default is ${{ github.token }}
    # Used to specify a package manager for caching in the default directory. Supported values: npm, yarn, pnpm
    cache: # optional
    # Deprecated. Use node-version instead. Will not be supported after October 1, 2019
    version: # optional
    
    Getting started with a workflow
To help you get started, this guide shows you some basic examples. For the full GitHub Actions documentation on workflows, see "Configuring workflows."
Customizing when workflow runs are triggered
Set your workflow to run on push events to the main and release/* branches
on:
  push:
    branches:
    - main
    - release/*
Set your workflow to run on pull_request events that target the main branch
on:
  pull_request:
    branches:
    - main
Set your workflow to run every day of the week from Monday to Friday at 2:00 UTC
on:
  schedule:
  - cron: "0 2 * * 1-5"
For more information, see "Events that trigger workflows."
Manually running a workflow
To manually run a workflow, you can configure your workflow to use the workflow_dispatch event. This enables a "Run workflow" button on the Actions tab.
on:
  workflow_dispatch:
For more information, see "Manually running a workflow."
Running your jobs on different operating systems
GitHub Actions provides hosted runners for Linux, Windows, and macOS.
To set the operating system for your job, specify the operating system using runs-on:
jobs:
  my_job:
    name: deploy to staging
    runs-on: ubuntu-18.04
The available virtual machine types are:
ubuntu-latest, ubuntu-18.04, or ubuntu-16.04
windows-latest or windows-2019
macos-latest or macos-10.15
For more information, see "Virtual environments for GitHub Actions."
Using an action
Actions are reusable units of code that can be built and distributed by anyone on GitHub. You can find a variety of actions in GitHub Marketplace, and also in the official Actions repository.
To use an action, you must specify the repository that contains the action. We also recommend that you specify a Git tag to ensure you are using a released version of the action.
- name: Setup Node
  uses: actions/setup-node@v1
  with:
    node-version: '10.x'
For more information, see "Workflow syntax for GitHub Actions."
Running a command
You can run commands on the job's virtual machine.
- name: Install Dependencies
  run: npm install
For more information, see "Workflow syntax for GitHub Actions."
Running a job across a matrix of operating systems and runtime versions
You can automatically run a job across a set of different values, such as different versions of code libraries or operating systems.
For example, this job uses a matrix strategy to run across 3 versions of Node and 3 operating systems:
jobs:
  test:
    name: Test on node ${{ matrix.node_version }} and ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        node_version: ['8', '10', '12']
        os: [ubuntu-latest, windows-latest, macOS-latest]
    steps:
    - uses: actions/checkout@v1
    - name: Use Node.js ${{ matrix.node_version }}
      uses: actions/setup-node@v1
      with:
        node-version: ${{ matrix.node_version }}
    - name: npm install, build and test
      run: |
        npm install
        npm run build --if-present
        npm test
For more information, see "Workflow syntax for GitHub Actions."
Running steps or jobs conditionally
GitHub Actions supports conditions on steps and jobs using data present in your workflow context.
For example, to run a step only as part of a push and not in a pull_request, you can specify a condition in the if: property based on the event name:
steps:
- run: npm publish
  if: github.event == 'push'
For more information, see "Contexts and expression syntax for GitHub Actions."
# *note-("*/*")'#"$_-
*/*
*
2020-09-30T12:07:41.1045079Z ##[section]Starting: Component Detection (auto-injected by policy)
2020-09-30T12:07:41.1053947Z ==============================================================================
2020-09-30T12:07:41.1054559Z Task         : Component Governance Detection
2020-09-30T12:07:41.1055070Z Description  : Include with your build to enable automatic Component Governance detection.
2020-09-30T12:07:41.1055512Z Version      : 0.2020924.1
2020-09-30T12:07:41.1055879Z Author       : Microsoft Corporation
2020-09-30T12:07:41.1056507Z Help         : Please contact OpenSourceEngSupport@microsoft.com if you run into problems or have questions with this task. See http://aka.ms/cgdocs for more information.
2020-09-30T12:07:41.1057200Z ==============================================================================
2020-09-30T12:07:41.1237169Z ##[error]No such file or directory
2020-09-30T12:07:41.1249841Z ##[section]Finishing: Component Detection (auto-injected by policy)*"/*
# @@ -1,31 +1,20 @@
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
                    
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/passes/weight_layout/transpose_2d.h"
#include "core/codegen/passes/weight_layout/vertical_stripes_2d.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"
#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/providers/nuphar/mti_x86/math/matmul_ops.h"
#include <tvm/ir_pass.h>
namespace onnxruntime {pull_sender_origin-request-docs.java
namespace nuphar {update_test_cfg-clean-echo-clean_build.java
 TODO: 
namespace onnxruntime {pull_sender_origin-request-docs.java
namespace nuphar {update_test_cfg-clean-echo-clean_build.java
namespace onnxruntime {
namespace nuphar {
/ TODO: 
/
static bool MatMul_weights2D(
static  string_MatMul_weights2D(
    ONNX_NAMESPACE::TensorProto_DataType proto_type,
    const tvm::Tensor& A,
    const tvm::Tensor& B,
    const tvm::Tensor& 1,
    const tvm::Tensor& 2,
    const std::string& initializer_name,
    NupharCodeGenCtx& ctx_codegen,
    tvm::Tensor& Y,
    tvm::Tensor& 3,
    const std::string& name = "matmul_weights2d") {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);
  // optimizations for B being 2D weights
  // The 2D weight is marshalled with stripe_width.
  // This should be 2x nature vector width
  int stripe_width = 8;
  int block_size = 32;
  onnxruntime::CodeGenTargetX86* target =
      dynamic_cast<onnxruntime::CodeGenTargetX86*>(ctx_codegen.GetCodeGenHandle()->codegen_target);
  if (nullptr != target) {
    stripe_width = 2 * target->NaturalVectorWidth(B->dtype.bits());
  }
  // align A, B to multiple of block size
  const auto& A_shape = A->shape;
  tvm::Expr A0_size = tvm_codegen::SizeToDimension(A_shape, -1);
  auto A0_roundup = tvm_codegen::RoundUp(A0_size, block_size);
  tvm::Expr A1_size = tvm_codegen::SizeFromDimension(A_shape, -1);
  auto A1_roundup = tvm_codegen::RoundUp(A1_size, block_size);
  bool A0_need_pad = !tvm::ir::Equal(A0_roundup, A0_size);
  bool A1_need_pad = !tvm::ir::Equal(A1_roundup, A1_size);
  const auto& B_shape = B->shape;
  tvm::Expr B0_size = tvm_codegen::SizeToDimension(B_shape, 1);
  auto B0_roundup = tvm_codegen::RoundUp(B0_size, block_size);
  tvm::Expr B1_size = tvm_codegen::SizeFromDimension(B_shape, 1);
  auto B1_roundup = tvm_codegen::RoundUp(B1_size, block_size);
  bool B1_need_pad = !tvm::ir::Equal(B1_roundup, B1_size);
  ORT_ENFORCE(tvm::ir::Equal(A1_roundup, B0_roundup));
  // Currently only support padding in B1, as it's free with memory marshalling
  if (A0_need_pad || A1_need_pad || B1_need_pad)
    return false;
  auto layout_key = tvm_codegen::WeightLayoutVerticalStripe2D::GetKey(proto_type, stripe_width);
  auto B_unmarshalled = ctx_nuphar->ApplyWeightLayout(layout_key, initializer_name, B, false);
  ORT_ENFORCE(B_unmarshalled->op.as<tvm::ComputeOpNode>());
  tvm::Array<tvm::Expr> Y_shape;
  for (size_t d = 0; d < A->shape.size() - 1; ++d)
    Y_shape.push_back(A->shape[d]);
  Y_shape.push_back(B->shape[1]);
  auto k = tvm::reduce_axis(tvm::Range(0, A1_size), "k");
  Y = tvm::compute(
      Y_shape,
      [&](const tvm::Array<tvm::Var>& idx) {
        tvm::Array<tvm::Expr> A_indices;
        for (size_t d = 0; d < idx.size() - 1; ++d)
          A_indices.push_back(idx[d]);
        A_indices.push_back(k);
        return tvm::sum(A(A_indices) * B_unmarshalled(k, idx[idx.size() - 1]), {k});
      },
      name);
  return true;
}
static bool MatMulF32ExternCPU(
    tvm::Tensor A,
    tvm::Tensor B,
    tvm::Tensor& Y,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);
  // try to fuse tranpose in MatMul input with MatMul
  auto find_transposed_input = [&ctx_nuphar](const tvm::Tensor& t, std::vector<int32_t>& cumulated_permute) {
    tvm::Tensor out = t;
    int64_t rank = gsl::narrow<int64_t>(t->shape.size());
    std::vector<int64_t> default_node_perm(rank);
    cumulated_permute.resize(rank);
    for (int64_t i = 0; i < rank; ++i) {
      cumulated_permute[i] = gsl::narrow<int32_t>(i);
      default_node_perm[i] = rank - i - 1;
    }
    for (const Node* root_node = ctx_nuphar->FindNode(out);
         root_node != nullptr && root_node->OpType() == "Transpose";
         root_node = ctx_nuphar->FindNode(out)) {
      ProtoHelperNodeContext ctx
      (root_node);
      OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
      auto perm = info.GetAttrsOrDefault("perm", default_node_perm);
      std::vector<int32_t> updated_cumulated_permute = cumulated_permute;
      for (int64_t dst_dim = 0; dst_dim < rank; ++dst_dim) {
        auto src_dim = tvm_codegen::HandleNegativeAxis(perm[cumulated_permute[dst_dim]], rank);
        updated_cumulated_permute[dst_dim] = gsl::narrow<int32_t>(src_dim);
      }
      cumulated_permute = updated_cumulated_permute;
      // op corresponding to node should be Transpose
      auto op = out->op.as<tvm::ComputeOpNode>();
      ORT_ENFORCE(op != nullptr);
      ORT_ENFORCE(op->InputTensors().size() == 1);
      out = op->InputTensors()[0];
    }
    return out;
  };
  std::vector<int32_t> permute_A;
  std::vector<int32_t> permute_B;
  const std::vector<int32_t>* p_permute_A = nullptr;
  const std::vector<int32_t>* p_permute_B = nullptr;
  tvm::Tensor root_A = find_transposed_input(A, permute_A);
  tvm::Tensor root_B = find_transposed_input(B, permute_B);
  bool transA = false;
  if (A->shape.size() == B->shape.size() && A->shape.size() >= 2) {
    // currently only fuse Transpose into MatMul when rank(A) == rank(B)
    // make sure no broadcasting in MatMul
    bool no_broadcast = true;
    for (size_t i = 0; i < A->shape.size() - 2; ++i) {
      if (!tvm::ir::Equal(A->shape[i], B->shape[i])) {
        no_broadcast = false;
        break;
      }
    }
    if (no_broadcast) {
      if (CanPermuteBeFusedInMatMul(permute_A)) {
        if (A != root_A)
          transA = true;
        A = root_A;
        p_permute_A = &permute_A;
      }
      if (CanPermuteBeFusedInMatMul(permute_B)) {
        B = root_B;
        p_permute_B = &permute_B;
      }
    }
  }
  const auto& B_name = node.InputDefs()[1]->Name();
  if (ctx_nuphar->IsInitializer(B_name) && B->shape.size() == 2) {
    if (A->shape.size() == 1) {
      return nuphar::GemmExternCpu(A, B, Y, transA, false, B_name);
    } else {
      // matmul with initializer, using transpose weights
      auto layout_key = tvm_codegen::WeightLayoutTranspose2D::GetKey(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      auto actual_B = ctx_nuphar->ApplyWeightLayout(layout_key, B_name, B, true);
      return nuphar::GemmExternCpu(A, actual_B, Y, transA, true, B_name);
    }
  } else {
    return nuphar::MatMulExternCpu(A, B, Y, p_permute_A, p_permute_B, node.Name() + "_matmul_extern");
  }
}
Status
NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(MatMul)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);
  auto proto_type = TensorProtoDataType(node.InputDefs()[1]);
  tvm::Tensor Y;
  auto& A = inputs[0]
  auto& B = inputs[1]
  }
  // if B is 2D initializer, use vertical stripe layout
  const std::string& input_1_name = node.InputDefs()[1]->Name();
  if (ShapeRank(node.InputDefs()[1]) == 2 && ctx_nuphar->IsInitializer(input_1_name)) {
    if (MatMul_weights2D(proto_type, A, B, input_1_name, *ctx_nuphar, Y)) {
      outputs.push_back(Y);
      return Status::OK();
   
## Build Pipeline Status
|System|CPU|GPU|EPs|
|---|---|---|---|label=OS)](https://dev.azure.com/onnxruntime/onnxruntime/_build-latest
## Data/Telemetry
Windows distributions of this project may collect usage data and send it to Microsoft to help improve our products and services.
See the [privacy statement](docs/Privacy.md) for more details.
## Contributions and Feedback
We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).
For feature requests or bug reports, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).
For general discussion or questions, please use [Github Discussions](https://github.com/microsoft/onnxruntime/discussions).
## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
## License
This project is licensed under the [MIT License](LICENSE).
