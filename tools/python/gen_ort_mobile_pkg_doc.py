from util import reduced_build_config_parser
from util.ort_format_model.operator_type_usage_processors import GloballyAllowedTypesOpTypeImplFilter

enable_type_reduction = True
config_file = r'D:\src\github\ort.vs19\tools\ci_build\github\android\mobile_package.required_operators.config'
output_file = r'D:\src\github\ort.vs19\docs\ORTMobilePackageOperatorTypeSupport.md'
required_ops, op_type_impl_filter = reduced_build_config_parser.parse_config(config_file, enable_type_reduction)

with open(output_file, 'w') as out:
    out.write('# ONNX Runtime Mobile Pre-Built Package Operator and Type Support\n\n')

    # Description
    out.write('## Supported operators and types\n\n')
    out.write('The supported operators and types are based on what is required to support the float32 and quantized '
              'versions of popular models. The full list of input models used to detemine this list is available '
              '[here](https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/android/mobile_package.required_operators.readme.txt)')  # noqa
    out.write('\n\n')

    # Globally supported types
    out.write('## Supported data input types\n\n')
    assert(op_type_impl_filter.__class__ is GloballyAllowedTypesOpTypeImplFilter)
    global_types = op_type_impl_filter.global_type_list()
    for type in sorted(global_types):
        out.write('  - {}\n'.format(type))
    out.write('\n')
    out.write('NOTE: Operators that are used to manipulate dimensions and indicies will support int32 and int64.\n\n')

    domain_op_opsets = []
    for domain in sorted(required_ops.keys()):
        op_opsets = {}
        domain_op_opsets.append((domain, op_opsets))
        for opset in sorted(required_ops[domain].keys()):
            str_opset = str(opset)
            for op in required_ops[domain][opset]:
                op_with_domain = '{}:{}'.format(domain, op)
                if op_with_domain not in op_opsets:
                    op_opsets[op_with_domain] = []

                op_opsets[op_with_domain].append(str_opset)

    out.write('## Supported Operators\n\n')
    out.write('|Operator|Opsets|\n')
    out.write('|--------|------|\n')
    for domain, op_opsets in domain_op_opsets:
        out.write('|**{}**||\n'.format(domain))
        for op in sorted(op_opsets.keys()):
            out.write('|{}|{}|\n'.format(op, ', '.join(op_opsets[op])))
        out.write('|||\n')



