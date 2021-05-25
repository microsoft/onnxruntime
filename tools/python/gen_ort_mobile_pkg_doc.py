import argparse
import os
import pathlib
from util import reduced_build_config_parser
from util.ort_format_model.operator_type_usage_processors import GloballyAllowedTypesOpTypeImplFilter


def generate_docs(output_file, required_ops, op_type_impl_filter):
    with open(output_file, 'w') as out:
        out.write('# ONNX Runtime Mobile Pre-Built Package Operator and Type Support\n\n')

        # Description
        out.write('## Supported operators and types\n\n')
        out.write('The supported operators and types are based on what is required to support float32 and quantized '
                  'versions of popular models. The full list of input models used to determine this list is available '
                  '[here](https://github.com/microsoft/onnxruntime/blob/master/tools/ci_build/github/android/mobile_package.required_operators.readme.txt)')  # noqa
        out.write('\n\n')

        # Globally supported types
        out.write('## Supported data input types\n\n')
        assert(op_type_impl_filter.__class__ is GloballyAllowedTypesOpTypeImplFilter)
        global_types = op_type_impl_filter.global_type_list()
        for type in sorted(global_types):
            out.write('  - {}\n'.format(type))
        out.write('\n')
        out.write('NOTE: Operators used to manipulate dimensions and indices will support int32 and int64.\n\n')

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


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(
        description='ONNX Runtime Mobile Pre-Built Package Operator and Type Support Documentation Generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_config_path = \
        pathlib.Path(os.path.join(script_dir, '../ci_build/github/android/mobile_package.required_operators.config')
                     ).resolve()

    default_output_path = \
        pathlib.Path(os.path.join(script_dir, '../../docs/ORTMobilePackageOperatorTypeSupport.md')).resolve()

    parser.add_argument('--config_path', help='Path to build configuration used to generate package.', required=False,
                        type=pathlib.Path, default=default_config_path)

    parser.add_argument('--output_path', help='output markdown file path', required=False,
                        type=pathlib.Path, default=default_output_path)

    args = parser.parse_args()
    config_file = args.config_path.resolve(strict=True)  # must exist so strict=True
    output_path = args.output_path.resolve()

    enable_type_reduction = True
    required_ops, op_type_impl_filter = reduced_build_config_parser.parse_config(config_file, enable_type_reduction)
    generate_docs(output_path, required_ops, op_type_impl_filter)


if __name__ == '__main__':
    main()
