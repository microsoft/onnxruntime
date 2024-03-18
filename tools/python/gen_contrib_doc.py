#!/usr/bin/env python

# This file is copied and adapted from https://github.com/onnx/onnx repository.
# There was no copyright statement on the file at the time of copying.

import argparse
import os
import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set, Text, Tuple  # noqa: F401

import numpy as np  # type: ignore
from onnx import AttributeProto, FunctionProto  # noqa: F401

import onnxruntime.capi.onnxruntime_pybind11_state as rtpy
from onnxruntime.capi.onnxruntime_pybind11_state import schemadef  # noqa: F401
from onnxruntime.capi.onnxruntime_pybind11_state.schemadef import OpSchema

ONNX_ML = not bool(os.getenv("ONNX_ML") == "0")
ONNX_DOMAIN = "onnx"
ONNX_ML_DOMAIN = "onnx-ml"

if ONNX_ML:
    ext = "-ml.md"
else:
    ext = ".md"


def display_number(v):  # type: (int) -> Text
    if OpSchema.is_infinite(v):
        return "&#8734;"
    return str(v)


def should_render_domain(domain, domain_filter):  # type: (Text) -> bool
    if domain in (ONNX_DOMAIN, ONNX_ML_DOMAIN) or domain == "" or domain == "ai.onnx.ml":
        return False

    if domain_filter and domain not in domain_filter:
        return False

    return True


def format_name_with_domain(domain, schema_name):  # type: (Text, Text) -> Text
    if domain:
        return f"{domain}.{schema_name}"
    else:
        return schema_name


def format_name_with_version(schema_name, version):  # type: (Text, Text) -> Text
    return f"{schema_name}-{version}"


def display_attr_type(v):  # type: (OpSchema.AttrType) -> Text
    assert isinstance(v, OpSchema.AttrType)
    s = str(v)
    s = s[s.rfind(".") + 1 :].lower()
    if s[-1] == "s":
        s = "list of " + s
    return s


def display_domain(domain):  # type: (Text) -> Text
    if domain:
        return f"the '{domain}' operator set"
    else:
        return "the default ONNX operator set"


def display_domain_short(domain):  # type: (Text) -> Text
    if domain:
        return domain
    else:
        return "ai.onnx (default)"


def display_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = "Changelog" + ext
    name_with_ver = f"{name}-{version}"
    return f'<a href="{changelog_md}#{name_with_ver}">{name_with_ver}</a>'


def display_function_version_link(name, version):  # type: (Text, int) -> Text
    changelog_md = "FunctionsChangelog" + ext
    name_with_ver = f"{name}-{version}"
    return f'<a href="{changelog_md}#{name_with_ver}">{name_with_ver}</a>'


def get_attribute_value(attr):  # type: (AttributeProto) -> Any
    if attr.HasField("f"):
        return attr.f
    elif attr.HasField("i"):
        return attr.i
    elif attr.HasField("s"):
        return attr.s
    elif attr.HasField("t"):
        return attr.t
    elif attr.HasField("g"):
        return attr.g
    elif len(attr.floats):
        return list(attr.floats)
    elif len(attr.ints):
        return list(attr.ints)
    elif len(attr.strings):
        return list(attr.strings)
    elif len(attr.tensors):
        return list(attr.tensors)
    elif len(attr.graphs):
        return list(attr.graphs)
    else:
        raise ValueError(f"Unsupported ONNX attribute: {attr}")


def display_schema(schema, versions):  # type: (OpSchema, Sequence[OpSchema]) -> Text
    s = ""

    # doc
    schemadoc = schema.doc
    if schemadoc:
        s += "\n"
        s += "\n".join("  " + line for line in schemadoc.lstrip().splitlines())
        s += "\n"

    # since version
    s += "\n#### Version\n"
    if schema.support_level == OpSchema.SupportType.EXPERIMENTAL:
        s += "\nNo versioning maintained for experimental ops."
    else:
        s += (
            "\nThis version of the operator has been "
            + ("deprecated" if schema.deprecated else "available")
            + f" since version {schema.since_version}"
        )
        s += f" of {display_domain(schema.domain)}.\n"
        if len(versions) > 1:
            # TODO: link to the Changelog.md
            s += "\nOther versions of this operator: {}\n".format(
                ", ".join(
                    format_name_with_version(format_name_with_domain(v.domain, v.name), v.since_version)
                    for v in versions[:-1]
                )
            )

    # If this schema is deprecated, don't display any of the following sections
    if schema.deprecated:
        return s

    # attributes
    attribs = schema.attributes
    if attribs:
        s += "\n#### Attributes\n\n"
        s += "<dl>\n"
        for _, attr in sorted(attribs.items()):
            # option holds either required or default value
            opt = ""
            if attr.required:
                opt = "required"
            elif hasattr(attr, "default_value") and attr.default_value.name:
                default_value = get_attribute_value(attr.default_value)

                def format_value(value):  # type: (Any) -> Text
                    if isinstance(value, float):
                        value = np.round(value, 5)
                    if isinstance(value, (bytes, bytearray)) and sys.version_info[0] == 3:  # noqa: YTT201
                        value = value.decode("utf-8")
                    return str(value)

                if isinstance(default_value, list):
                    default_value = [format_value(val) for val in default_value]
                else:
                    default_value = format_value(default_value)
                opt = f"default is {default_value}"

            s += "<dt><tt>{}</tt> : {}{}</dt>\n".format(
                attr.name, display_attr_type(attr.type), f" ({opt})" if opt else ""
            )
            s += f"<dd>{attr.description}</dd>\n"
        s += "</dl>\n"

    # inputs
    s += "\n#### Inputs"
    if schema.min_input != schema.max_input:
        s += f" ({display_number(schema.min_input)} - {display_number(schema.max_input)})"
    s += "\n\n"

    inputs = schema.inputs
    if inputs:
        s += "<dl>\n"
        for inp in inputs:
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == inp.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == inp.option:
                if inp.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += f"<dt><tt>{inp.name}</tt>{option_str} : {inp.typeStr}</dt>\n"
            s += f"<dd>{inp.description}</dd>\n"

    s += "</dl>\n"

    # outputs
    s += "\n#### Outputs"
    if schema.min_output != schema.max_output:
        s += f" ({display_number(schema.min_output)} - {display_number(schema.max_output)})"
    s += "\n\n"
    outputs = schema.outputs
    if outputs:
        s += "<dl>\n"
        for output in outputs:
            option_str = ""
            if OpSchema.FormalParameterOption.Optional == output.option:
                option_str = " (optional)"
            elif OpSchema.FormalParameterOption.Variadic == output.option:
                if output.isHomogeneous:
                    option_str = " (variadic)"
                else:
                    option_str = " (variadic, heterogeneous)"
            s += f"<dt><tt>{output.name}</tt>{option_str} : {output.typeStr}</dt>\n"
            s += f"<dd>{output.description}</dd>\n"

    s += "</dl>\n"

    # type constraints
    s += "\n#### Type Constraints"
    s += "\n\n"
    typecons = schema.type_constraints
    if typecons:
        s += "<dl>\n"
        for type_constraint in typecons:
            allowed_types = type_constraint.allowed_type_strs
            allowed_type_str = ""
            if len(allowed_types) > 0:
                allowed_type_str = allowed_types[0]
            for allowedType in allowed_types[1:]:  # noqa: N806
                allowed_type_str += ", " + allowedType
            s += f"<dt><tt>{type_constraint.type_param_str}</tt> : {allowed_type_str}</dt>\n"
            s += f"<dd>{type_constraint.description}</dd>\n"
        s += "</dl>\n"

    return s


def display_function(function, versions, domain=ONNX_DOMAIN):  # type: (FunctionProto, List[int], Text) -> Text
    s = ""

    if domain:
        domain_prefix = f"{ONNX_ML_DOMAIN}."
    else:
        domain_prefix = ""

    # doc
    if function.doc_string:
        s += "\n"
        s += "\n".join("  " + line for line in function.doc_string.lstrip().splitlines())
        s += "\n"

    # since version
    s += "\n#### Version\n"
    s += f"\nThis version of the function has been available since version {function.since_version}"
    s += f" of {display_domain(domain_prefix)}.\n"
    if len(versions) > 1:
        s += "\nOther versions of this function: {}\n".format(
            ", ".join(
                display_function_version_link(domain_prefix + function.name, v)
                for v in versions
                if v != function.since_version
            )
        )

    # inputs
    s += "\n#### Inputs"
    s += "\n\n"
    if function.input:
        s += "<dl>\n"
        for input in function.input:
            s += f"<dt>{input}; </dt>\n"
        s += "<br/></dl>\n"

    # outputs
    s += "\n#### Outputs"
    s += "\n\n"
    if function.output:
        s += "<dl>\n"
        for output in function.output:
            s += f"<dt>{output}; </dt>\n"
        s += "<br/></dl>\n"

        # attributes
    if function.attribute:
        s += "\n#### Attributes\n\n"
        s += "<dl>\n"
        for attr in function.attribute:
            s += f"<dt>{attr};<br/></dt>\n"
        s += "</dl>\n"

    return s


def support_level_str(level):  # type: (OpSchema.SupportType) -> Text
    return "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""


# def function_status_str(status=OperatorStatus.Value("EXPERIMENTAL")):  # type: ignore
#     return \
#         "<sub>experimental</sub> " if status == OperatorStatus.Value('EXPERIMENTAL') else ""  # type: ignore


def main(output_path: str, domain_filter: [str]):
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        fout.write("## Contrib Operator Schemas\n")
        fout.write(
            "*This file is automatically generated from the registered contrib operator schemas by "
            "[this script](https://github.com/microsoft/onnxruntime/blob/main/tools/python/gen_contrib_doc.py).\n"
            "Do not modify directly.*\n"
        )

        # domain -> support level -> name -> [schema]
        index = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]

        for schema in rtpy.get_all_operator_schema():
            index[schema.domain][int(schema.support_level)][schema.name].append(schema)

        fout.write("\n")

        # Preprocess the Operator Schemas
        # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
        operator_schemas = (
            list()
        )  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
        exsting_ops = set()  # type: Set[Text]
        for domain, _supportmap in sorted(index.items()):
            if not should_render_domain(domain, domain_filter):
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
            s = f"* {display_domain_short(domain)}\n"
            fout.write(s)

            for _, namemap in supportmap:
                for n, schema, versions in namemap:  # noqa: B007
                    s = f'  * {support_level_str(schema.support_level)}<a href="#{format_name_with_domain(domain, n)}">{format_name_with_domain(domain, n)}</a>\n'
                    fout.write(s)

        fout.write("\n")

        for domain, supportmap in operator_schemas:
            s = f"## {display_domain_short(domain)}\n"
            fout.write(s)

            for _, namemap in supportmap:
                for op_type, schema, versions in namemap:
                    # op_type
                    s = (
                        '### {}<a name="{}"></a><a name="{}">**{}**'
                        + (" (deprecated)" if schema.deprecated else "")
                        + "</a>\n"
                    ).format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, op_type),
                        format_name_with_domain(domain, op_type.lower()),
                        format_name_with_domain(domain, op_type),
                    )

                    s += display_schema(schema, versions)

                    s += "\n\n"

                    fout.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Runtime Contrib Operator Documentation Generator")
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Filter to specified domains. " "e.g. `--domains com.microsoft com.microsoft.nchwc`",  # noqa: ISC001
    )
    parser.add_argument(
        "--output_path",
        help="output markdown file path",
        type=pathlib.Path,
        required=True,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "ContribOperators.md"),
    )
    args = parser.parse_args()
    output_path = args.output_path.resolve()

    main(output_path, args.domains)
