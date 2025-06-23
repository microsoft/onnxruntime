#include "python/onnxruntime_pybind_state_common.h"
#include "core/framework/kernel_registry.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

// Use the nanobind namespace
namespace nb = nanobind;

namespace onnxruntime {
namespace python {

void addGlobalSchemaFunctions(nanobind::module_& m) {
  m.def(
      "get_all_operator_schema", []() -> const std::vector<ONNX_NAMESPACE::OpSchema> {
        return ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();
      },
      "Return a vector of OpSchema all registered operators");
  m.def(
      "get_all_opkernel_def", []() -> const std::vector<onnxruntime::KernelDef> {
        std::vector<onnxruntime::KernelDef> result;

        std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> factories = {
            onnxruntime::CPUProviderFactoryCreator::Create(0),
#ifdef USE_CUDA
            []() {
              OrtCUDAProviderOptions provider_options{};
              return CudaProviderFactoryCreator::Create(&provider_options);
            }(),
#endif
#ifdef USE_ROCM
            []() {
              OrtROCMProviderOptions provider_options;
              return onnxruntime::RocmProviderFactoryCreator::Create(&provider_options);
            }(),
#endif
#ifdef USE_DNNL
            onnxruntime::DnnlProviderFactoryCreator::Create(1),
#endif
#ifdef USE_OPENVINO
            []() {
              ProviderOptions provider_options_map;
              SessionOptions session_options;
              return onnxruntime::OpenVINOProviderFactoryCreator::Create(&provider_options_map, &session_options);
            }(),
#endif
#ifdef USE_TENSORRT
            onnxruntime::TensorrtProviderFactoryCreator::Create(0),
#endif
#ifdef USE_NV
            onnxruntime::NvProviderFactoryCreator::Create(0),
#endif
#ifdef USE_MIGRAPHX
            onnxruntime::MIGraphXProviderFactoryCreator::Create(0),
#endif
#ifdef USE_VITISAI
            onnxruntime::VitisAIProviderFactoryCreator::Create(ProviderOptions{}),
#endif
#ifdef USE_ACL
            onnxruntime::ACLProviderFactoryCreator::Create(false),
#endif
#ifdef USE_ARMNN
            onnxruntime::ArmNNProviderFactoryCreator::Create(0),
#endif
#ifdef USE_DML
            []() {
              ConfigOptions config_options{};
              return onnxruntime::DMLProviderFactoryCreator::Create(config_options, 0, false, false, false);
            }(),
#endif
#ifdef USE_NNAPI
            onnxruntime::NnapiProviderFactoryCreator::Create(0, std::optional<std::string>()),
#endif
#ifdef USE_VSINPU
            onnxruntime::VSINPUProviderFactoryCreator::Create(),
#endif
#ifdef USE_RKNPU
            onnxruntime::RknpuProviderFactoryCreator::Create(),
#endif
#ifdef USE_COREML
            onnxruntime::CoreMLProviderFactoryCreator::Create(ProviderOptions{}),
#endif
#ifdef USE_XNNPACK
            onnxruntime::XnnpackProviderFactoryCreator::Create(ProviderOptions{}, nullptr),
#endif
#ifdef USE_CANN
            []() {
              OrtCANNProviderOptions provider_options{};
              return CannProviderFactoryCreator::Create(&provider_options);
            }(),
#endif
        };

        for (const auto& f : factories) {
          auto kernel_registry = f->CreateProvider()->GetKernelRegistry();
          for (const auto& m : kernel_registry->GetKernelCreateMap()) {
            result.emplace_back(*(m.second.kernel_def));
          }
        }

        return result;
      },
      "Return a vector of KernelDef for all registered OpKernels");
}

// Update function signature to use nanobind::module_
void addOpKernelSubmodule(nanobind::module_& m) {
  auto opkernel = m.def_submodule("opkernel");
  opkernel.doc() = "OpKernel submodule";

  // Use nb::class_
  nb::class_<onnxruntime::KernelDef> kernel_def(opkernel, "KernelDef");
  kernel_def.def_prop_ro("op_name", &onnxruntime::KernelDef::OpName)
      .def_prop_ro("domain", &onnxruntime::KernelDef::Domain)
      .def_prop_ro("provider", &onnxruntime::KernelDef::Provider)
      .def_prop_ro("version_range",
                             [](const onnxruntime::KernelDef& kernelDef) -> std::pair<int, int> {
                               return kernelDef.onnxruntime::KernelDef::SinceVersion();
                             })
      .def_prop_ro("type_constraints",
                             [](const onnxruntime::KernelDef& kernelDef) -> std::unordered_map<std::string, std::vector<std::string>> {
                               std::unordered_map<std::string, std::vector<std::string>> result;
                               const auto& tempResult = kernelDef.TypeConstraints();
                               for (const auto& tc : tempResult) {
                                 result[tc.first] = std::vector<std::string>();
                                 for (const auto& dt : tc.second) {
                                   result[tc.first].emplace_back(onnxruntime::DataTypeImpl::ToString(dt));
                                 }
                               }
                               return result;
                             });
}

// Update function signature to use nanobind::module_
void addOpSchemaSubmodule(nanobind::module_& m) {
  auto schemadef = m.def_submodule("schemadef");
  schemadef.doc() = "Schema submodule";

  nb::class_<ONNX_NAMESPACE::OpSchema> op_schema(schemadef, "OpSchema");
  op_schema.def_prop_ro("file", &ONNX_NAMESPACE::OpSchema::file)
      .def_prop_ro("line", &ONNX_NAMESPACE::OpSchema::line)
      .def_prop_ro("support_level", &ONNX_NAMESPACE::OpSchema::support_level)
      .def_prop_ro("doc", &ONNX_NAMESPACE::OpSchema::doc, nb::rv_policy::reference)
      .def_prop_ro("since_version", &ONNX_NAMESPACE::OpSchema::since_version)
      .def_prop_ro("deprecated", &ONNX_NAMESPACE::OpSchema::deprecated)
      .def_prop_ro("domain", &ONNX_NAMESPACE::OpSchema::domain)
      .def_prop_ro("name", &ONNX_NAMESPACE::OpSchema::Name)
      .def_prop_ro("min_input", &ONNX_NAMESPACE::OpSchema::min_input)
      .def_prop_ro("max_input", &ONNX_NAMESPACE::OpSchema::max_input)
      .def_prop_ro("min_output", &ONNX_NAMESPACE::OpSchema::min_output)
      .def_prop_ro("max_output", &ONNX_NAMESPACE::OpSchema::max_output)
      .def_prop_ro("attributes", &ONNX_NAMESPACE::OpSchema::attributes)
      .def_prop_ro("inputs", &ONNX_NAMESPACE::OpSchema::inputs)
      .def_prop_ro("outputs", &ONNX_NAMESPACE::OpSchema::outputs)
      .def_prop_ro(
          "has_type_and_shape_inference_function",
          &ONNX_NAMESPACE::OpSchema::has_type_and_shape_inference_function)
      .def_prop_ro(
          "type_constraints", &ONNX_NAMESPACE::OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) {
        return v == std::numeric_limits<int>::max();
      });

  nb::class_<ONNX_NAMESPACE::OpSchema::Attribute>(op_schema, "Attribute")
      .def_ro("name", &ONNX_NAMESPACE::OpSchema::Attribute::name)
      .def_ro("description", &ONNX_NAMESPACE::OpSchema::Attribute::description)
      .def_ro("type", &ONNX_NAMESPACE::OpSchema::Attribute::type)
      .def_prop_ro(
          "_default_value",
          // Use nb::bytes
          [](ONNX_NAMESPACE::OpSchema::Attribute* attr) -> nb::bytes {
            std::string out;
            attr->default_value.SerializeToString(&out);
            return nb::bytes(out.data(), out.size());
          })
      .def_ro("required", &ONNX_NAMESPACE::OpSchema::Attribute::required);

  nb::class_<ONNX_NAMESPACE::OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam")
      .def_ro(
          "type_param_str", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::type_param_str)
      .def_ro("description", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::description)
      .def_ro(
          "allowed_type_strs",
          &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::allowed_type_strs);

  nb::enum_<ONNX_NAMESPACE::OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption")
      .value("Single", ONNX_NAMESPACE::OpSchema::Single)
      .value("Optional", ONNX_NAMESPACE::OpSchema::Optional)
      .value("Variadic", ONNX_NAMESPACE::OpSchema::Variadic);

  nb::class_<ONNX_NAMESPACE::OpSchema::FormalParameter>(op_schema, "FormalParameter")
      .def_prop_ro("name", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetName)
      .def_prop_ro("types", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypes)
      .def_prop_ro("typeStr", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypeStr)
      .def_prop_ro(
          "description", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetDescription)
      .def_prop_ro("option", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetOption)
      .def_prop_ro(
          "isHomogeneous", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetIsHomogeneous);

  nb::enum_<ONNX_NAMESPACE::AttributeProto::AttributeType>(op_schema, "AttrType")
      .value("FLOAT", ONNX_NAMESPACE::AttributeProto::FLOAT)
      .value("INT", ONNX_NAMESPACE::AttributeProto::INT)
      .value("STRING", ONNX_NAMESPACE::AttributeProto::STRING)
      .value("TENSOR", ONNX_NAMESPACE::AttributeProto::TENSOR)
      .value("SPARSE_TENSOR", ONNX_NAMESPACE::AttributeProto::SPARSE_TENSOR)
      .value("GRAPH", ONNX_NAMESPACE::AttributeProto::GRAPH)
      .value("FLOATS", ONNX_NAMESPACE::AttributeProto::FLOATS)
      .value("INTS", ONNX_NAMESPACE::AttributeProto::INTS)
      .value("STRINGS", ONNX_NAMESPACE::AttributeProto::STRINGS)
      .value("TENSORS", ONNX_NAMESPACE::AttributeProto::TENSORS)
      .value("SPARSE_TENSORS", ONNX_NAMESPACE::AttributeProto::SPARSE_TENSORS)
      .value("GRAPHS", ONNX_NAMESPACE::AttributeProto::GRAPHS);

  nb::enum_<ONNX_NAMESPACE::OpSchema::SupportType>(op_schema, "SupportType")
      .value("COMMON", ONNX_NAMESPACE::OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", ONNX_NAMESPACE::OpSchema::SupportType::EXPERIMENTAL);
}
}  // namespace python
}  // namespace onnxruntime