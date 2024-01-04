#pragma once
#include "common.h"

//Need to file bugs for failing tests and add to reason. Before that happens, default reasons will be used.
static const std::string disabledTestDefaultReason =
  "Model not working on CPU and GPU. Please file bug and replace this reason message.";
static const std::string disabledGpuTestDefaultReason =
  "Model not working on GPU. Please file bug and replace this reason message.";

// {"model test name", "reason for why it is happening and bug filed for it."}
std::unordered_map<std::string, std::string> disabledTests({
  // Disabled cases common to both CPU&GPU (no _CPU/_GPU suffix):

  // Tier 3 models
  {                                                           "mxnet_arcface_opset8",      disabledTestDefaultReason                                                                                     },
  {                                  "XGBoost_XGClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                         "XGBoost_XGClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                                   "XGBoost_XGClassifier_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                      "XGBoost_XGClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                            "scikit_SVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                                   "scikit_SVC_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                                             "scikit_SVC_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                                "scikit_SVC_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {             "scikit_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {          "scikit_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                         "scikit_RandomForestClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                "scikit_RandomForestClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                          "scikit_RandomForestClassifier_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                                         "scikit_Nu_SVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                                "scikit_Nu_SVC_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                                          "scikit_Nu_SVC_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                             "scikit_Nu_SVC_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {              "scikit_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                           "scikit_Normalizer_LinearSVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                    "scikit_LogisticRegression_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                              "scikit_LogisticRegression_OpenML_31_credit_opset7",                                                                                           disabledTestDefaultReason},
  {                              "scikit_LogisticRegression_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                 "scikit_LogisticRegression_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                      "scikit_LinearSVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                                      "scikit_LabelEncoder_OpenML_3_chess_opset7",                                                                                           disabledTestDefaultReason},
  {                                         "scikit_LabelEncoder_BikeSharing_opset7",                                                                                           disabledTestDefaultReason},
  {            "scikit_Imputer_LogisticRegression_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  { "scikit_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7",                                                                                           disabledTestDefaultReason},
  {    "scikit_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  { "scikit_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                     "scikit_GradientBoostingClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {            "scikit_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                      "scikit_GradientBoostingClassifier_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {         "scikit_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                         "scikit_DictVectorizer_sklearn_load_Iris_missing_opset7",                                                                                           disabledTestDefaultReason},
  {                               "scikit_DictVectorizer_sklearn_load_digits_opset7",                                                                                           disabledTestDefaultReason},
  {                     "scikit_DictVectorizer_sklearn_load_diabetes_missing_opset7",                                                                                           disabledTestDefaultReason},
  {                                  "scikit_DictVectorizer_OpenML_31_credit_opset7",                                                                                           disabledTestDefaultReason},
  {       "scikit_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7",                                                                                           disabledTestDefaultReason},
  {          "scikit_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                       "scikit_DictVectorizer_LinearSVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {            "scikit_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7",                                                                                           disabledTestDefaultReason},
  {     "scikit_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7",                                                                                           disabledTestDefaultReason},
  {                         "scikit_DecisionTreeClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                "scikit_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                          "scikit_DecisionTreeClassifier_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {             "scikit_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {           "scikit_Binarization_DecisionTreeClassifier_OpenML_1492_plants_opset7",                                                                                           disabledTestDefaultReason},
  {"scikit_Binarization_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                         "libsvm_Nu_SVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                                "libsvm_Nu_SVC_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                                          "libsvm_Nu_SVC_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                             "libsvm_Nu_SVC_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                                   "coreml_VGG16_ImageNet_opset7",                                                                                           disabledTestDefaultReason},
  {                                            "coreml_SVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                                   "coreml_SVC_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                                             "coreml_SVC_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                                "coreml_SVC_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                              "coreml_SqueezeNet_ImageNet_opset7",                                                                                           disabledTestDefaultReason},
  {             "coreml_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {          "coreml_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                                "coreml_Resnet50_ImageNet_opset7",                                                                                           disabledTestDefaultReason},
  {                         "coreml_RandomForestClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                "coreml_RandomForestClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                          "coreml_RandomForestClassifier_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {             "coreml_RandomForestClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {              "coreml_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                           "coreml_Normalizer_LinearSVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                    "coreml_LogisticRegression_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                              "coreml_LogisticRegression_OpenML_31_credit_opset7",                                                                                           disabledTestDefaultReason},
  {                              "coreml_LogisticRegression_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                 "coreml_LogisticRegression_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                      "coreml_LinearSVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                             "coreml_LinearSVC_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                                       "coreml_LinearSVC_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {                          "coreml_LinearSVC_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                             "coreml_Inceptionv3_ImageNet_opset7",                                                                                           disabledTestDefaultReason},
  { "coreml_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7",                                                                                           disabledTestDefaultReason},
  {    "coreml_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  { "coreml_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                     "coreml_GradientBoostingClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {            "coreml_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                      "coreml_GradientBoostingClassifier_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {         "coreml_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                "coreml_GradientBoostingClassifier_Criteo_opset7",                                                                                           disabledTestDefaultReason},
  {                             "coreml_GradientBoostingClassifier_BingClick_opset7",                                                                                           disabledTestDefaultReason},
  {                         "coreml_DictVectorizer_sklearn_load_Iris_missing_opset7",                                                                                           disabledTestDefaultReason},
  {                               "coreml_DictVectorizer_sklearn_load_digits_opset7",                                                                                           disabledTestDefaultReason},
  {                     "coreml_DictVectorizer_sklearn_load_diabetes_missing_opset7",                                                                                           disabledTestDefaultReason},
  {                                  "coreml_DictVectorizer_OpenML_31_credit_opset7",                                                                                           disabledTestDefaultReason},
  {       "coreml_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7",                                                                                           disabledTestDefaultReason},
  {          "coreml_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                       "coreml_DictVectorizer_LinearSVC_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {            "coreml_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7",                                                                                           disabledTestDefaultReason},
  {     "coreml_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7",                                                                                           disabledTestDefaultReason},
  {                         "coreml_DecisionTreeClassifier_sklearn_load_wine_opset7",                                                                                           disabledTestDefaultReason},
  {                "coreml_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7",                                                                                           disabledTestDefaultReason},
  {                          "coreml_DecisionTreeClassifier_OpenML_312_scene_opset7",                                                                                           disabledTestDefaultReason},
  {             "coreml_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7",                                                                                           disabledTestDefaultReason},
  {                                                  "coreml_AgeNet_ImageNet_opset7",                                                                                           disabledTestDefaultReason},

 // GPU specific cases:

  // ONNX zoo models
  {                                                          "mask_rcnn_opset10_GPU",
   "Bug 31005388: mask_rcnn opset 10 onnx zoo model fails to evaluate on DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/31005388"                                                        },
  {                                                        "faster_rcnn_opset10_GPU",
   "Bug 31005511: Failed to extract tensor data from evaluate result of faster_rcnn opset 10 model in DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/31005511"                           },
 // ONNX model zoo's int8/qdq models generally do not work on CPUs that lack 8-bit instructions.
  {                                                         "YOLOv3_12_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                            "VGG_16_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                               "SSD_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                      "ShuffleNet_v2_qdq_opset12",                                                                                           disabledTestDefaultReason},
  {                                                     "ShuffleNet_v2_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                           "ResNet50_qdq_opset12",                                                                                           disabledTestDefaultReason},
  {                                                          "ResNet50_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                   "MobileNet_v2_1_0_qdq_opset12",                                                                                           disabledTestDefaultReason},
  {                                                  "MobileNet_v2_1_0_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                       "Inception_1_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                             "Faster_R_CNN_R_50_FPN_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                        "BERT_Squad_int8_opset12",                                                                                           disabledTestDefaultReason},
  {                                                 "EfficientNet_Lite4_qdq_opset11",                                                                                           disabledTestDefaultReason},
  {                                                "EfficientNet_Lite4_int8_opset11",                                                                                           disabledTestDefaultReason},

 // Tier 2 models
  {                                               "fp16_test_tiny_yolov2_opset7_GPU",
   "Bug 31005780: Result of fp16_test_tiny_yolov2_opset7 and fp16_coreml_FNS_Candy_opset7 models on DirectML aren't as accurate as on CPU https://microsoft.visualstudio.com/OS/_workitems/edit/31005780"},
  {                                                    "fp16_tiny_yolov2_opset8_GPU",
   "Bug 31005780: Result of fp16_test_tiny_yolov2_opset7 and fp16_coreml_FNS_Candy_opset7 models on DirectML aren't as accurate as on CPU https://microsoft.visualstudio.com/OS/_workitems/edit/31005780"},
  {                                               "fp16_coreml_FNS_Candy_opset7_GPU",
   "Bug 31005780: Result of fp16_test_tiny_yolov2_opset7 and fp16_coreml_FNS_Candy_opset7 models on DirectML aren't as accurate as on CPU https://microsoft.visualstudio.com/OS/_workitems/edit/31005780"},
  {                                           "mlperf_ssd_mobilenet_300_opset10_GPU",
   "Bug 31005624: mlperf_ssd_mobilenet_300 opset 10 model fails to evaluate in DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/31005624"                                                  },
});

/*
    model name -> (adapter name regex, skipped test reason)
*/
std::unordered_map<std::string, std::pair<std::string, std::string>> disabledGpuAdapterTests({
  // e.g. {"fp16_inception_v1_opset7_GPU", std::make_pair("NVIDIA", "Bug 31144419: Results of fp16_inception_v1 opset7 and opset8 aren't accurate enough on AMD Radeon VII & Intel(R) UHD Graphics 630 & NVIDIA https://microsoft.visualstudio.com/OS/_workitems/edit/31144419")},
  //      {"candy_opset9", std::make_pair("(Intel\\(R\\) (UHD )?Graphics)|(Adreno)", "Bug 31652854: Results of candy_opset9 aren't accurate enough on Intel Graphics and Qualcomm Adreno 685 https://microsoft.visualstudio.com/OS/_workitems/edit/31652854")},
});

/*
    Override the default tolerances for these test cases (can be tailored to only CPU or GPU with suffix).
    test name -> absolute difference sampleTolerance
*/
std::unordered_map<std::string, double> sampleTolerancePerTests({
  {"fp16_inception_v1_opset7_GPU",      0.005},
  {"fp16_inception_v1_opset8_GPU",      0.005},
  {            "candy_opset9_GPU", 0.00150000}, // Intel(R) UHD Graphics 630 (29.20.100.9020) AP machine has inaccurate GPU results for FNS Candy opset 9 https://microsoft.visualstudio.com/OS/_workitems/edit/30696168/
  { "fp16_tiny_yolov2_opset8_GPU",   0.109000}, // Intel(R) UHD Graphics 630 (29.20.100.9020) AP machine has inaccurate GPU results for FNS Candy opset 9 https://microsoft.visualstudio.com/OS/_workitems/edit/30696168/
});
