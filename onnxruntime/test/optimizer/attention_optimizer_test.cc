#include "gtest/gtest.h"
#include "gtest/gtest.h"

#include "asserts.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "dummy_graph_transformer.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/not_where_fusion.h"
#include "core/optimizer/attention_fusion.h"

namespace onnxruntime {
namespace test {
    TEST(AttentionOptimizerTest, BertModel)
    {
        auto model_uri = "testdata/transform/fusion/attention_int32_mask.onnx";
        std::shared_ptr<Model> model;
        ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                                DefaultLoggingManager().DefaultLogger())
                        .IsOK());
        Graph& graph = model->MainGraph();
        // define compatible providers and register transformer
        const std::unordered_set<std::string> cpu_cuda_rocm_eps = {onnxruntime::kCpuExecutionProvider,
                                                                 onnxruntime::kCudaExecutionProvider,
                                                                 onnxruntime::kRocmExecutionProvider};
        std::unique_ptr<GraphTransformer> attention_transformer = std::make_unique<AttentionFusion>(cpu_cuda_rocm_eps);
        bool modified = false;
        ASSERT_STATUS_OK(attention_transformer->Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
        auto op_counts = CountOpsInGraph(graph);
        EXPECT_EQ(op_counts["Attention"], 1);  // result should contain Attention op
    }

    TEST(AttentionOptimizerTest, DistilBertModel)
    {
        auto model_uri = "testdata/transform/fusion/attention_distilbert.onnx";
        std::shared_ptr<Model> model;
        ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
                                DefaultLoggingManager().DefaultLogger())
                        .IsOK());
        Graph& graph = model->MainGraph();
        // define compatible providers and register transformer
        const std::unordered_set<std::string> cpu_cuda_rocm_eps = {onnxruntime::kCpuExecutionProvider,
                                                                 onnxruntime::kCudaExecutionProvider,
                                                                 onnxruntime::kRocmExecutionProvider};
        std::unique_ptr<GraphTransformer> attention_transformer = std::make_unique<AttentionFusion>(cpu_cuda_rocm_eps);
        bool modified = false;
        ASSERT_STATUS_OK(attention_transformer->Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
        auto op_counts = CountOpsInGraph(graph);
        EXPECT_EQ(op_counts["Attention"], 1);
    }
}
}