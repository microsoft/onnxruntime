#!/usr/bin/env python

# This file is copied and adapted from https://github.com/onnx/onnx repository.
# There was no copyright statement on the file at the time of copying.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import io
import os
import sys

import numpy as np  # type: ignore

import onnxruntime as rt
import onnxruntime.capi.onnxruntime_pybind11_state as rtpy 
from onnxruntime.capi.onnxruntime_pybind11_state import schemadef 
#from onnx import defs, FunctionProto, helper, OperatorStatus
from onnxruntime.capi.onnxruntime_pybind11_state.schemadef import OpSchema  #, ONNX_DOMAIN, ONNX_ML_DOMAIN
#from onnx.backend.test.case import collect_snippets
#from onnx.backend.sample.ops import collect_sample_implementations
from typing import Any, Text, Sequence, Dict, List, Type, Set, Tuple


#SNIPPETS = collect_snippets()
#SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = not bool(os.getenv('ONNX_ML') == '0')
ONNX_DOMAIN = "onnx"
ONNX_ML_DOMAIN = "onnx-ml"

if ONNX_ML:
    ext = '-ml.md'
else:
    ext = '.md'


def display_number(v):  # type: (int) -> Text
    if OpSchema.is_infinite(v):
        return '&#8734;'
    return Text(v)


def should_render_domain(domain):  # type: (Text) -> bool
    # if domain == ONNX_ML_DOMAIN and not ONNX_ML:
    #     return False
    # elif ONNX_ML and domain != ONNX_ML_DOMAIN:
    #     return False
    if domain == ONNX_DOMAIN or domain == '':
        return False
    return True


def format_name_with_domain(domain, schema_name):  # type: (Text, Text) -> Text
    if domain:
        return '{}.{}'.format(domain, schema_name)
    else:
        return schema_name


def display_attr_type(v):  # type: (OpSchema.AttrType) -> Text
    assert isinstance(v, OpSchema.AttrType)
    s = Text(v)
    s = s[s.rfind('.') + 1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s


def display_domain(domain):  # type: (Text) -> Text
    if domain:
        return "the '{}' operator set".format(domain)
    else:
        return "the default ONNX operator set"


def display_domain_short(domain):  # type: (Text) -> Text
    if domain:
        return domain
    else:
        return 'ai.onnx (default)'


def display_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = 'Changelog' + ext
    name_with_ver = '{}-{}'.format(name, version)
    return '<a href="{}#{}">{}</a>'.format(changelog_md, name_with_ver, name_with_ver)


def display_function_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = 'FunctionsChangelog' + ext
    name_with_ver = '{}-{}'.format(name, version)
    return '<a href="{}#{}">{}</a>'.format(changelog_md, name_with_ver, name_with_ver)


def display_schema(schema, versions):  # type: (OpSchema, Sequence[OpSchema]) -> Text
    s = ''

    # doc
    if schema.doc:
        s += '\n'
        s += '\n'.join('  ' + line
                       for line in schema.doc.lstrip().splitlines())
        s += '\n'

    # since version
    s += '\n#### Version\n'
    if schema.support_level == OpSchema.SupportType.EXPERIMENTAL:
        s += '\nNo versioning maintained for experimental ops.'
    else:
        s += '\nThis version of the operator has been ' + ('deprecated' if schema.deprecated else 'available') + ' since version {}'.format(schema.since_version)
        s += ' of {}.\n'.format(display_domain(schema.domain))
        if len(versions) > 1:
            # TODO: link to the Changelog.md
            s += '\nOther versions of this operator: {}\n'.format(
                ', '.join(display_version_link(format_name_with_domain(v.domain, v.name),
                                               v.since_version) for v in versions[:-1]))

    # If this schema is deprecated, don't display any of the following sections
    if schema.deprecated:
        return s

    # attributes
    if schema.attributes:
        s += '\n#### Attributes\n\n'
        s += '<dl>\n'
        for _, attr in sorted(schema.attributes.items()):
            # option holds either required or default value
            opt = ''
            if attr.required:
                opt = 'required'
            elif hasattr(attr, 'default_value') and attr.default_value.name:
                default_value = helper.get_attribute_value(attr.default_value)

                def format_value(value):  # type: (Any) -> Text
                    if isinstance(value, float):
                        value = np.round(value, 5)
                    if isinstance(value, (bytes, bytearray)) and sys.version_info[0] == 3:
                        value = value.decode('utf-8')
                    return str(value)

                if isinstance(default_value, list):
                    default_value = [format_value(val) for val in default_value]
                else:
                    default_value = format_value(default_value)
                opt = 'default is {}'.format(default_value)

            s += '<dt><tt>{}</tt> : {}{}</dt>\n'.format(
                attr.name,
                display_attr_type(attr.type),
                ' ({})'.format(opt) if opt else '')
            s += '<dd>{}</dd>\n'.format(attr.description)
        s += '</dl>\n'

    # inputs
    s += '\n#### Inputs'
    if schema.min_input != schema.max_input:
        s += ' ({} - {})'.format(display_number(schema.min_input),
                                 display_number(schema.max_input))
    s += '\n\n'
    if schema.inputs:
        s += '<dl>\n'
        for input in schema.inputs:
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == input.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == input.option:
                if input.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += '<dt><tt>{}</tt>{} : {}</dt>\n'.format(input.name, option_str, input.typeStr)
            s += '<dd>{}</dd>\n'.format(input.description)
        s += '</dl>\n'

    # outputs
    s += '\n#### Outputs'
    if schema.min_output != schema.max_output:
        s += ' ({} - {})'.format(display_number(schema.min_output),
                                 display_number(schema.max_output))
    s += '\n\n'

    if schema.outputs:
        s += '<dl>\n'
        for output in schema.outputs:
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == output.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == output.option:
                if output.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += '<dt><tt>{}</tt>{} : {}</dt>\n'.format(output.name, option_str, output.typeStr)
            s += '<dd>{}</dd>\n'.format(output.description)
        s += '</dl>\n'

    # type constraints
    s += '\n#### Type Constraints'
    s += '\n\n'
    if schema.type_constraints:
        s += '<dl>\n'
        for type_constraint in schema.type_constraints:
            allowedTypes = type_constraint.allowed_type_strs
            allowedTypeStr = ''
            if (len(allowedTypes) > 0):
                allowedTypeStr = allowedTypes[0]
            for allowedType in allowedTypes[1:]:
                allowedTypeStr += ', ' + allowedType
            s += '<dt><tt>{}</tt> : {}</dt>\n'.format(
                type_constraint.type_param_str, allowedTypeStr)
            s += '<dd>{}</dd>\n'.format(type_constraint.description)
        s += '</dl>\n'

    return s


def display_function(function, versions, domain=ONNX_DOMAIN):  # type: (FunctionProto, List[int], Text) -> Text
    s = ''

    if domain:
        domain_prefix = '{}.'.format(ONNX_ML_DOMAIN)
    else:
        domain_prefix = ''

    # doc
    if function.doc_string:
        s += '\n'
        s += '\n'.join('  ' + line
                       for line in function.doc_string.lstrip().splitlines())
        s += '\n'

    # since version
    s += '\n#### Version\n'
    s += '\nThis version of the function has been available since version {}'.format(function.since_version)
    s += ' of {}.\n'.format(display_domain(domain_prefix))
    if len(versions) > 1:
        s += '\nOther versions of this function: {}\n'.format(
            ', '.join(display_function_version_link(domain_prefix + function.name, v) for v in versions if v != function.since_version))

    # inputs
    s += '\n#### Inputs'
    s += '\n\n'
    if function.input:
        s += '<dl>\n'
        for input in function.input:
            s += '<dt>{}; </dt>\n'.format(input)
        s += '<br/></dl>\n'

    # outputs
    s += '\n#### Outputs'
    s += '\n\n'
    if function.output:
        s += '<dl>\n'
        for output in function.output:
            s += '<dt>{}; </dt>\n'.format(output)
        s += '<br/></dl>\n'

        # attributes
    if function.attribute:
        s += '\n#### Attributes\n\n'
        s += '<dl>\n'
        for attr in function.attribute:
            s += '<dt>{};<br/></dt>\n'.format(attr)
        s += '</dl>\n'

    return s


def support_level_str(level):  # type: (OpSchema.SupportType) -> Text
    return \
        "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""


# def function_status_str(status=OperatorStatus.Value("EXPERIMENTAL")):  # type: ignore
#     return \
#         "<sub>experimental</sub> " if status == OperatorStatus.Value('EXPERIMENTAL') else ""  # type: ignore


def main(args):  # type: (Type[Args]) -> None
    # with io.open(args.changelog, 'w', newline='') as fout:
    #     fout.write('## Operator Changelog\n')
    #     fout.write(
    #         "*This file is automatically generated from the\n"
    #         "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
    #         "            Do not modify directly and instead edit operator definitions.*\n")

    #     # domain -> version -> [schema]
    #     dv_index = defaultdict(lambda: defaultdict(list))  # type: Dict[Text, Dict[int, List[OpSchema]]]
    #     for schema in defs.get_all_schemas_with_history():
    #         dv_index[schema.domain][schema.since_version].append(schema)

    #     fout.write('\n')

    #     for domain, versionmap in sorted(dv_index.items()):
    #         if not should_render_domain(domain):
    #             continue

    #         s = '# {}\n'.format(display_domain_short(domain))

    #         for version, unsorted_schemas in sorted(versionmap.items()):
    #             s += '## Version {} of {}\n'.format(version, display_domain(domain))
    #             for schema in sorted(unsorted_schemas, key=lambda s: s.name):
    #                 name_with_ver = '{}-{}'.format(format_name_with_domain(domain, schema.name),
    #                                                schema.since_version)
    #                 s += ('### <a name="{}"></a>**{}**' + (' (deprecated)' if schema.deprecated else '') + '</a>\n').format(name_with_ver, name_with_ver)
    #                 s += display_schema(schema, [schema])
    #                 s += '\n'

    #         fout.write(s)

    # with io.open(args.fn_changelog, 'w', newline='') as fout:
    #     fout.write('## Function Changelog\n')
    #     fout.write(
    #         "*This file is automatically generated from the\n"
    #         "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
    #         "            Do not modify directly and instead edit function definitions.*\n")

    #     if ONNX_ML:
    #         all_functions = defs.get_functions(ONNX_ML_DOMAIN)
    #     else:
    #         all_functions = defs.get_functions('')

    #     changelog_versionmap = defaultdict(list)  # type: Dict[int, List[FunctionProto]]
    #     for fn_name, functions in sorted(all_functions.items()):
    #         for func in functions:
    #             changelog_versionmap[func.since_version].append(func)

    #     if ONNX_ML:
    #         s = '## {}\n'.format(ONNX_ML_DOMAIN)
    #         domain_display_name = ONNX_ML_DOMAIN
    #         domain_prefix = '{}.'.format(ONNX_ML_DOMAIN)
    #     else:
    #         s = '# ai.onnx (default)\n'
    #         domain_display_name = 'ai.onnx (default)'
    #         domain_prefix = ''
    #     fout.write(s)

    #     for version, function_list in sorted(changelog_versionmap.items()):
    #         s = ""
    #         for function in function_list:
    #             s += '## Version {} of domain {}\n'.format(version, domain_display_name)
    #             name_with_ver = '{}-{}'.format(domain_prefix
    #                                            + fn_name, function.since_version)
    #             s += '### <a name="{}"></a>**{}**</a>\n'.format(name_with_ver, name_with_ver)
    #             available_versions = [func.since_version for func in all_functions[function.name]]
    #             s += display_function(function, available_versions, domain_prefix)
    #             s += '\n'
    #         fout.write(s)

    with io.open(args.output, 'w', newline='', encoding="utf-8") as fout:
        fout.write('## Operator Schemas\n')
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n")

        # domain -> support level -> name -> [schema]
        index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
        for schema in rtpy.get_all_operator_schema():
            index[schema.domain][int(schema.support_level)][schema.name].append(schema)

        fout.write('\n')

        # Preprocess the Operator Schemas
        # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
        operator_schemas = list()  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
        exsting_ops = set()  # type: Set[Text]
        for domain, _supportmap in sorted(index.items()):
            if not should_render_domain(domain):
                continue

            processed_supportmap = list()
            for _support, _namemap in sorted(_supportmap.items()):
                processed_namemap = list()
                for n, unsorted_versions in sorted(_namemap.items()):
                    versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                    schema = versions[-1]
                    if schema.name in exsting_ops:
                        continue
                    exsting_ops.add(schema.name)
                    processed_namemap.append((n, schema, versions))
                processed_supportmap.append((_support, processed_namemap))
            operator_schemas.append((domain, processed_supportmap))

        # Table of contents
        for domain, supportmap in operator_schemas:
            s = '* {}\n'.format(display_domain_short(domain))
            fout.write(s)

            for _, namemap in supportmap:
                for n, schema, versions in namemap:
                    s = '  * {}<a href="#{}">{}</a>\n'.format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, n),
                        format_name_with_domain(domain, n))
                    fout.write(s)

        fout.write('\n')

        for domain, supportmap in operator_schemas:
            s = '## {}\n'.format(display_domain_short(domain))
            fout.write(s)

            for _, namemap in supportmap:
                for op_type, schema, versions in namemap:
                    # op_type
                    s = ('### {}<a name="{}"></a><a name="{}">**{}**' + (' (deprecated)' if schema.deprecated else '') + '</a>\n').format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, op_type),
                        format_name_with_domain(domain, op_type.lower()),
                        format_name_with_domain(domain, op_type))

                    s += display_schema(schema, versions)

                    s += '\n\n'

                    # if op_type in SNIPPETS:
                    #     s += '#### Examples\n\n'
                    #     for summary, code in sorted(SNIPPETS[op_type]):
                    #         s += '<details>\n'
                    #         s += '<summary>{}</summary>\n\n'.format(summary)
                    #         s += '```python\n{}\n```\n\n'.format(code)
                    #         s += '</details>\n'
                    #         s += '\n\n'
                    # if op_type.lower() in SAMPLE_IMPLEMENTATIONS:
                    #     s += '#### Sample Implementation\n\n'
                    #     s += '<details>\n'
                    #     s += '<summary>{}</summary>\n\n'.format(op_type)
                    #     s += '```python\n{}\n```\n\n'.format(SAMPLE_IMPLEMENTATIONS[op_type.lower()])
                    #     s += '</details>\n'
                    #     s += '\n\n'

                    fout.write(s)

    # with io.open(args.function_output, 'w', newline='') as fout:
    #     fout.write('## Functions\n')
    #     fout.write(
    #         "*This file is automatically generated from the\n"
    #         "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
    #         "            Do not modify directly and instead edit function definitions.*\n")

    #     if ONNX_ML:
    #         all_functions = defs.get_functions(ONNX_ML_DOMAIN)
    #     else:
    #         all_functions = defs.get_functions('')

    #     if all_functions:
    #         if ONNX_ML:
    #             s = '## {}\n'.format(ONNX_ML_DOMAIN)
    #             domain_prefix = '{}.'.format(ONNX_ML_DOMAIN)
    #         else:
    #             s = '## ai.onnx (default)\n'
    #             domain_prefix = ''
    #         fout.write(s)

    #         existing_functions = set()  # type: Set[Text]
    #         for function_name, functions in sorted(all_functions.items()):
    #             for function in sorted(functions, key=lambda s: s.since_version, reverse=True):
    #                 if function.name in existing_functions:
    #                     continue
    #                 existing_functions.add(function.name)
    #                 s = '  * {}<a href="#{}">{}</a>\n'.format(
    #                     function_status_str(function.status),
    #                     domain_prefix + function.name, domain_prefix + function.name)
    #                 fout.write(s)

    #             fout.write('\n')

    #         fout.write('\n\n')

    #         for function_name, functions in sorted(all_functions.items()):
    #             available_versions = [func.since_version for func in functions]
    #             function = sorted(functions, key=lambda s: s.since_version, reverse=True)[0]
    #             s = '### {}<a name="{}"></a><a name="{}">**{}**</a>\n'.format(
    #                 function_status_str(function.status),
    #                 domain_prefix + function.name, domain_prefix + function.name.lower(),
    #                 domain_prefix + function.name)

    #             s += display_function(function, available_versions, domain_prefix)
    #             s += '\n\n'
    #             fout.write(s)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    docs_dir = os.path.join(base_dir, 'docs')

    class Args(object):
        output = os.path.join(docs_dir, 'Operators' + ext)
        # function_output = os.path.join(docs_dir, 'Functions' + ext)
        # changelog = os.path.join(docs_dir, 'Changelog' + ext)
        # fn_changelog = os.path.join(docs_dir, 'FunctionsChangelog' + ext)
    main(Args)
