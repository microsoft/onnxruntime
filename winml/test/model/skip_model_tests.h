#pragma once
#include "common.h"

//Need to file bugs for failing tests and add to reason. Before that happens, default reasons will be used.
static const std::string disabledTestDefaultReason = "Model not working on CPU and GPU. Please file bug and replace this reason message.";
static const std::string disabledGpuTestDefaultReason = "Model not working on GPU. Please file bug and replace this reason message.";
static const std::string disabledx86TestDefaultReason = "Model not working on x86. Please file bug and replace this reason message.";

// {"model test name", "reason for why it is happening and bug filed for it."}
std::unordered_map<std::string, std::string> disabledTests(
    {// Onnx zoo models
     {"test_bertsquad_opset8", disabledTestDefaultReason},
     {"test_bidaf_opset9", "Strings haven't been implemented in model testing yet. Need to file a bug."},

     // Tier 2 models
     {"coreml_VGG16_ImageNet_opset8", disabledTestDefaultReason},
     {"coreml_VGG16_ImageNet_opset9", disabledTestDefaultReason},
     {"coreml_Resnet50_opset9", disabledTestDefaultReason},
     {"coreml_inceptionv3_opset9", disabledTestDefaultReason},
     {"coreml_VGG16_ImageNet_opset10", disabledTestDefaultReason},
     {"coreml_Resnet50_opset10", disabledTestDefaultReason},
     {"coreml_inceptionv3_opset10", disabledTestDefaultReason},

     // Tier 3 models
     {"mxnet_arcface_opset8", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_RandomForestClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_RandomForestClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_Normalizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_LabelEncoder_OpenML_3_chess_opset7", disabledTestDefaultReason},
     {"scikit_LabelEncoder_BikeSharing_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_sklearn_load_Iris_missing_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_sklearn_load_digits_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_sklearn_load_diabetes_missing_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_Binarization_DecisionTreeClassifier_OpenML_1492_plants_opset7", disabledTestDefaultReason},
     {"scikit_Binarization_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_VGG16_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_SqueezeNet_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_Resnet50_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_Normalizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_Inceptionv3_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7", disabledTestDefaultReason},
     {"coreml_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_Criteo_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_BingClick_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_sklearn_load_Iris_missing_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_sklearn_load_digits_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_sklearn_load_diabetes_missing_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_AgeNet_ImageNet_opset7", disabledTestDefaultReason}
    });

std::unordered_map<std::string, std::string> disabledGpuTests(
    {
     // Onnx zoo models
     {"mask_rcnn_opset10", disabledGpuTestDefaultReason},
     {"faster_rcnn_opset10", disabledGpuTestDefaultReason},
     {"BERT_Squad_opset10", disabledGpuTestDefaultReason},

     // Tier 2 models
     {"fp16_inception_v1_opset7", disabledGpuTestDefaultReason},
     {"fp16_test_tiny_yolov2_opset7", "Result of evaluation isn't accurate enough. Please file bug"},
     {"fp16_coreml_FNS_Candy_opset7", "Result of evaluation isn't accurate enough. Please file bug"},
     {"fp16_inception_v1_opset8", disabledGpuTestDefaultReason},
     {"LSTM_Seq_lens_unpacked_opset9", disabledGpuTestDefaultReason},
     {"mlperf_ssd_mobilenet_300_opset10", disabledGpuTestDefaultReason}
    });

std::unordered_map<std::string, std::string> disabledx86Tests(
    {
     // Onnx zoo
     {"mask_rcnn_opset10", disabledx86TestDefaultReason},
     {"faster_rcnn_opset10", disabledx86TestDefaultReason},
     {"GPT2_LM_HEAD_opset10", disabledx86TestDefaultReason},
     {"GPT2_opset10", disabledx86TestDefaultReason},
     {"BERT_Squad_opset10", disabledx86TestDefaultReason},

     // Tier 2 Models
     {"test_vgg19_opset7", disabledx86TestDefaultReason},
     {"test_vgg19_opset8", disabledx86TestDefaultReason},
     {"coreml_VGG16_ImageNet_opset7", disabledx86TestDefaultReason},
     {"mlperf_ssd_resnet34_1200_opset10", disabledx86TestDefaultReason},
    });