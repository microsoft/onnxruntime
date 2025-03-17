# pylint: disable=C0103,C0415,R0912,R0913,R0914,R0915
"""
Automates the generation of ONNX operators.
"""
import importlib
import inspect
import keyword
import os
import re
import sys
import textwrap
from difflib import Differ

import numpy as np
import onnx
from onnx.backend.test.case.base import _Exporter
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.numpy_helper import to_array
from onnx.onnx_cpp2py_export.defs import SchemaError  # pylint: disable=E1101,E0611,E0401
from sphinx.util import logging


def get_template():
    try:
        from jinja2 import Template
    except ImportError:  # pragma no cover

        class Template:
            "Docstring template"

            def __init__(self, *args):
                pass

            def render(self, **context):
                "render"
                schemas = context["schemas"]
                rows = []
                for sch in schemas:
                    doc = sch.doc or ""
                    name = sch.name
                    if name is None:
                        raise RuntimeError("An operator must have a name.")
                    rows.extend([name, "=" * len(name), "", doc, ""])
                return "\n".join(rows)

    return Template


def _get_diff_template():
    template = get_template()
    return template(
        textwrap.dedent(
            """
        <div id="{{ div_name }}"></div>
        <link rel="stylesheet" type="text/css" href="../_static/diff2html.min.css" />
        <script type="text/javascript" src="../_static/diff2html-ui.min.js"></script>
        <script>
        const diffString = `
        --- a/{{ op_name }}{{ version1 }}
        +++ b/{{ op_name }}{{ version2 }}
        @@ -1 +1 @@
        {{ diff_content }}
        `;

        document.addEventListener('DOMContentLoaded', function () {
        var targetElement = document.getElementById('{{ div_name }}');
        var configuration = {
            drawFileList: true,
            fileListToggle: false,
            fileListStartVisible: false,
            fileContentToggle: false,
            matching: 'lines',
            outputFormat: 'line-by-line',
            synchronisedScroll: true,
            highlight: true,
            renderNothingWhenEmpty: false,
        };
        var diff2htmlUi = new Diff2HtmlUI(targetElement, diffString, configuration);
        diff2htmlUi.draw();
        diff2htmlUi.highlightCode();
        });
        </script>
        """
        ),
        autoescape=True,
    )


def _get_ops_template():
    template = get_template()
    return template(
        textwrap.dedent(
            """
        {% for sch in schemas %}

        .. tag-diff-insert.
        .. _l-onnx-op{{sch.domain.lower().replace(".", "-")}}-{{sch.name.lower()}}-{{str(sch.since_version)}}:

        {{format_name_with_domain(sch)}}
        {{'=' * len(format_name_with_domain(sch))}}

        **Version**

        * **name**: `{{sch.name}} (GitHub) <{{build_doc_url(sch)}}{{sch.name}}>`_
        * **domain**: **{% if sch.domain == '' %}main{% else %}{{sch.domain}}{% endif %}**
        * **since_version**: **{{sch.since_version}}**
        * **function**: {{sch.has_function}}
        * **support_level**: {{sch.support_level}}
        * **shape inference**: {{sch.has_type_and_shape_inference_function}}

        {% if sch.support_level == OpSchema.SupportType.EXPERIMENTAL %}
        No versioning maintained for experimental ops.
        {% else %}
        This version of the operator has been {% if
        sch.deprecated %}deprecated{% else %}available{% endif %}
        **since version {{sch.since_version}}{% if
        sch.domain %} of domain {{sch.domain}}{% endif %}**.
        {% if len(sch.versions) > 1 %}
        Other versions of this operator:
        {% for v in sch.version[:-1] %} {{v}} {% endfor %}
        {% endif %}
        {% endif %}

        **Summary**

        {{process_documentation(sch.doc)}}
        {% if sch.attributes %}

        **Attributes**

        {% for _, attr in sorted(sch.attributes.items())
        %}* **{{attr.name}} - {{str(attr.type).split('.')[-1]}}**{%
          if attr.required %} (required){% endif %} {%
          if attr.default_value %}{{clean_default_value(attr)}}{%
          endif %}: {{text_wrap(attr.description, 2)}}
        {% endfor %}
        {% endif %}
        {% if sch.inputs %}

        **Inputs**

        {% if sch.min_input != sch.max_input %}Between {{sch.min_input
        }} and {{sch.max_input}} inputs.
        {% endif %}
        {% for ii, inp in enumerate(sch.inputs) %}
        * **{{getname(inp, ii)}}**{{format_option(inp)}} - **{{inp.typeStr}}**:
        {{text_wrap(inp.description, 2)}}{% endfor %}
        {% endif %}
        {% if sch.outputs %}

        **Outputs**

        {% if sch.min_output != sch.max_output %}Between {{sch.min_output
        }} and {{sch.max_output}} outputs.
        {% endif %}
        {% for ii, out in enumerate(sch.outputs) %}
        * **{{getname(out, ii)}}**{{format_option(out)}} - **{{out.typeStr}}**:
        {{text_wrap(out.description, 2)}}{% endfor %}
        {% endif %}
        {% if sch.type_constraints %}

        **Type Constraints**

        {% for ii, type_constraint in enumerate(sch.type_constraints)
        %}* {{get_constraint(type_constraint, ii)}}:
        {{text_wrap(type_constraint.description, 2)}}
        {% endfor %}
        {% endif %}
        {% if get_onnx_example and is_last_schema(sch): %}

        **Examples**

        {% for example, code in get_onnx_example(sch.name).items(): %}

        **{{ example }}**

        ::

        {{ format_example(code) }}
        {% endfor %}
        {% endif %}
        {% endfor %}
    """
        ),
        autoescape=True,
    )


def _get_main_template():
    template = get_template()
    return template(
        textwrap.dedent(
            """
        .. _l-onnx-operators:

        {{ title }}
        {{ "=" * len(title) }}

        Lists out all the ONNX operators defined in onnxruntime.

        .. toctree::
            :hidden:

            {% for p in pages %}{{ os.path.split(p)[-1] }}
            {% endfor %}

        .. tabs::

            {% for t in tabs %}.. tab:: {{ t.domain_name }}
                {{ t.render(indent="        ") }}
            {% endfor %}
    """
        ),
        autoescape=True,
    )


def _clean_unicode(text):
    text = text.replace("&#34;", '"')
    text = text.replace("&#8212;", "-")
    text = text.replace("&#160;", " ")
    text = text.replace("&#39;", "'")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    return text


_template_diff = _get_diff_template()
_template_operator = _get_ops_template()
_template_main = _get_main_template()
__get_all_schemas_with_history = None


_attribute_conversion_functions = {
    onnx.AttributeProto.FLOAT: lambda att: np.float32(att.f),
    onnx.AttributeProto.FLOATS: lambda att: [np.float32(f) for f in att.floats],
    # AttributeProto.GRAPH(5)
    # AttributeProto.GRAPHS(10)
    onnx.AttributeProto.INT: lambda att: int(att.i),
    onnx.AttributeProto.INTS: lambda att: [int(i) for i in att.ints],
    # AttributeProto.SPARSE_TENSOR(11)
    # AttributeProto.SPARSE_TENSORS(12)
    onnx.AttributeProto.STRING: lambda att: att.s.decode("utf-8"),
    onnx.AttributeProto.STRINGS: lambda att: [s.decode("utf-8") for s in att.strings],
    onnx.AttributeProto.TENSOR: lambda att: to_array(att.t),
    # AttributeProto.TENSORS(9)
    # onnx.AttributeProto.TYPE_PROTO: lambda att: OnnxType(att.tp),
    # AttributeProto.TYPE_PROTOS(14)
}


def _populate__get_all_schemas_with_history():
    import onnxruntime.capi.onnxruntime_pybind11_state as rtpy

    get_schemas = rtpy.get_all_operator_schema or rtpy.get_all_opkernel_def

    schemas = get_schemas()
    res = {}
    for sch in schemas:
        domain, name = sch.domain, sch.name
        if domain in res and name in res[domain]:
            # already handled
            continue
        version = sch.since_version
        if domain not in res:
            res[domain] = {}
        if name not in res[domain]:
            res[domain][name] = {}
        res[domain][name][version] = sch
    return res


def _get_all_schemas_with_history():
    global __get_all_schemas_with_history  # noqa: PLW0603
    if __get_all_schemas_with_history is None:
        __get_all_schemas_with_history = _populate__get_all_schemas_with_history()
    return __get_all_schemas_with_history


def get_domain_list():
    """
    Returns the list of available domains.
    """
    return list(sorted(set(map(lambda s: s.domain, get_all_schemas_with_history()))))


def get_operator_schemas(op_name, version=None, domain=None):
    """
    Returns all schemas mapped to an operator name.
    :param op_name: name of the operator
    :param version: version
    :param domain: domain
    :return: list of schemas
    """
    if version == "last" and op_name is not None:
        if domain is not None:
            return [get_schema(op_name, domain=domain)]
    all_schemas = _get_all_schemas_with_history()
    if domain is None:
        domains = []
        for dom, ops in all_schemas.items():
            if op_name is None or op_name in ops:
                domains.append(dom)
    else:
        domains = [domain]

    # schemas
    sch = []
    for dom in domains:
        ops = all_schemas[dom]
        if op_name is None:
            for op, v in ops.items():
                if version is None:
                    sch.extend(v.values())
                elif version == "last" and (not dom or "onnx" in dom):
                    try:
                        sch.append(get_schema(op, domain=dom))
                    except SchemaError:  # pragma: no cover
                        sch.append(v[max(v)])
                elif version == "last":
                    sch.append(v[max(v)])
                else:
                    sch.append(v[version])
        elif op_name in ops:
            if version is None:
                sch.extend(ops[op_name].values())
            elif version in ops[op_name]:
                sch.append(ops[op_name][version])

    # sort
    vals = [(s.domain, s.name, -s.since_version, s) for s in sch]
    vals.sort()
    return [v[-1] for v in vals]


def get_rst_doc(
    folder,
    op_name=None,
    domain=None,
    version="last",
    clean=True,
    diff=False,
    example=False,
):
    """
    Returns a documentation in RST format
    for all :class:`OnnxOperator`.

    :param op_name: operator name of None for all
    :param domain: domain
    :param version: version, None for all, `'last'` for the most recent one
    :param clean: clean empty lines
    :param diff: highlights differences between two versions
    :param example: add example to the documentation
    :return: string
    The function relies on module `jinja2` or replaces it
    with a simple rendering if not present.
    """
    schemas = get_operator_schemas(op_name, domain=domain, version=version)

    # from onnx.backend.sample.ops import collect_sample_implementations
    # from onnx.backend.test.case import collect_snippets
    # SNIPPETS = collect_snippets()
    # SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
    def format_name_with_domain(sch):
        if version == "last":
            if sch.domain:
                return f"{sch.name} ({sch.domain})"
            return sch.name
        return f"{sch.name} - {sch.since_version}"

    def format_option(obj):
        opts = []
        if OpSchema.FormalParameterOption.Optional == obj.option:
            opts.append("optional")
        elif OpSchema.FormalParameterOption.Variadic == obj.option:
            opts.append("variadic")
        if getattr(obj, "isHomogeneous", False):
            opts.append("heterogeneous")
        if opts:
            return f" ({', '.join(opts)})"
        return ""

    def format_example(code):
        code = textwrap.indent(code, "    ")
        return code

    def get_constraint(const, ii):
        if const.type_param_str:
            name = const.type_param_str
        else:
            name = str(ii)
        name = f"**{name}** in ("
        if const.allowed_type_strs:
            text = ",\n  ".join(sorted(const.allowed_type_strs))
            name += "\n  " + text + "\n  )"
        return name

    def getname(obj, i):
        name = obj.name
        if len(name) == 0:
            return str(i)
        return name

    def process_documentation(doc):
        if doc is None:
            doc = ""
        if not isinstance(doc, str):
            raise TypeError(f"doc must be a string not {type(doc)!r} - {doc + 42!r}.")  # pragma: no cover
        doc = textwrap.dedent(doc)
        main_docs_url = "https://github.com/onnx/onnx/blob/master/"
        rep = {
            "[the doc](IR.md)": "`ONNX <{0}docs/IR.md>`_",
            "[the doc](Broadcasting.md)": "`Broadcasting in ONNX <{0}docs/Broadcasting.md>`_",
            "<dl>": "",
            "</dl>": "",
            "<dt>": "* ",
            "<dd>": "  ",
            "</dt>": "",
            "</dd>": "",
            "<tt>": "``",
            "</tt>": "``",
            "<br>": "\n",
        }
        for k, v in rep.items():
            doc = doc.replace(k, v.format(main_docs_url))
        move = 0
        lines = []
        for line in doc.split("\n"):
            if line.startswith("```"):
                if move > 0:
                    move -= 4
                    lines.append("\n")
                else:
                    lines.append("::\n")
                    move += 4
            elif move > 0:
                lines.append(" " * move + line)
            else:
                lines.append(line)
        return "\n".join(lines)

    def build_doc_url(sch):
        doc_url = "https://github.com/onnx/onnx/blob/main/docs/Operators"
        if "ml" in sch.domain:
            doc_url += "-ml"
        doc_url += ".md"
        doc_url += "#"
        if sch.domain not in (None, "", "ai.onnx"):
            doc_url += sch.domain + "."
        return doc_url

    def format_default_value(value):
        if isinstance(value, float):
            formatted = str(np.round(value, 5))
            # use default formatting, unless too long.
            if len(formatted) > 10:
                formatted = f"({value:e})"
            return formatted
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return str(value)

    def clean_default_value(attr):
        if isinstance(attr.default_value, str):
            raise TypeError(f"Unexpected type for {type(attr)} - {attr}.")
        if not attr.default_value.name:
            return ""
        default_value = onnx.helper.get_attribute_value(attr.default_value)
        if isinstance(default_value, onnx.AttributeProto) and hasattr(default_value, "default_value"):
            if attr.type in _attribute_conversion_functions:
                sval = _attribute_conversion_functions[attr.type](default_value)
                return f"(default is ``{sval!r}``)"

        if isinstance(default_value, list):
            sval = [format_default_value(val) for val in default_value]
        else:
            sval = format_default_value(default_value)
        return f"(default is ``{sval!r}``)"

    def text_wrap(text, indent):
        s = " " * indent
        lines = textwrap.wrap(text, initial_indent=s, subsequent_indent=s)
        return "\n".join(lines)

    fnwd = format_name_with_domain
    tmpl = _template_operator
    docs = tmpl.render(
        schemas=schemas,
        OpSchema=OpSchema,
        len=len,
        getattr=getattr,
        sorted=sorted,
        format_option=format_option,
        get_constraint=get_constraint,
        getname=getname,
        enumerate=enumerate,
        format_name_with_domain=fnwd,
        process_documentation=process_documentation,
        build_doc_url=build_doc_url,
        text_wrap=text_wrap,
        str=str,
        clean_default_value=clean_default_value,
        get_onnx_example=get_onnx_example if example else None,
        format_example=format_example,
        is_last_schema=is_last_schema,
    )
    docs = _clean_unicode(docs)

    d_links = {}
    for schema in schemas:
        sdom = schema.domain.replace(".", "-")
        d_links[schema.since_version] = f"l-onnx-op{sdom}-{schema.name.lower()}-{schema.since_version}"

    if diff:
        lines = docs.split("\n")
        new_lines = [""]
        for line_ in lines:
            line = line_.rstrip("\r\t ")
            if len(line) == 0 and len(new_lines[-1]) == 0:
                continue
            new_lines.append(line)
        docs = "\n".join(new_lines)
        docs, d_links_diff = _insert_diff(
            folder,
            docs,
            ".. tag-diff-insert.",
            op_name=op_name,
            version=version,
            domain=domain,
        )
        d_links.update(d_links_diff)

    if clean:
        lines = docs.split("\n")
        new_lines = [""]
        for line_ in lines:
            line = line_.rstrip("\r\t ")
            if len(line) == 0 and len(new_lines[-1]) == 0:
                continue
            new_lines.append(line)
        docs = "\n".join(new_lines)

    return docs, d_links


def _insert_diff(folder, docs, split=".. tag-diff-insert.", op_name=None, version=None, domain=None):
    """
    Splits a using `split`, insert HTML differences between pieces.
    The function relies on package `pyquickhelper`.
    """
    spl = docs.split(split)
    if len(spl) <= 1:
        return docs

    reg = re.compile("([A-Z][A-Za-z0-9_]*) - ([0-9]+)")

    d_links = {}
    pieces = [spl[0]]
    mds = []
    for i in range(1, len(spl)):
        spl1 = spl[i - 1].strip("\n ")
        spl2 = spl[i].strip("\n ")
        vers1 = reg.findall(spl1)
        vers2 = reg.findall(spl2)

        spl1 = spl1.split("**Examples**")[0].replace("`", "")
        spl2 = spl2.split("**Examples**")[0].replace("`", "")
        spl1 = spl1.split("**Summary**")[-1].strip("\n ")
        spl2 = spl2.split("**Summary**")[-1].strip("\n ")
        if len(spl1) < 5 or len(spl2) < 5:
            pieces.append(spl[i])
            continue
        if len(vers1) == 0:
            raise ValueError(f"Unable to find version {version!r} in\n{spl1}")
        if len(vers2) == 0:
            raise ValueError(f"Unable to find version {version!r} in\n{spl2}")
        v2 = vers2[0][1]
        v1 = vers1[0][1]

        if len(mds) == 0:
            mds.append((v1, textwrap.dedent(spl1.strip(" \n\r\t")).splitlines(keepends=True)))
        mds.append((v2, textwrap.dedent(spl2.strip(" \n\r\t")).splitlines(keepends=True)))

        if len(mds) > 1:
            pieces.extend([".. toctree::", ""])

        for di in range(len(mds) - 1):
            dj = len(mds) - 1

            v1, s1 = mds[di]
            v2, s2 = mds[dj]
            d = Differ()
            result = list(d.compare(s2, s1))
            raw = "".join(result)

            tmpl = _template_diff
            diff = tmpl.render(
                op_name=op_name,
                version1=v2,
                version2=v1,
                div_name=f"div_{op_name}_{i}",
                diff_content=raw,
            )
            diff = _clean_unicode(diff)

            title = f"{op_name} - {v2} vs {v1}"

            name = f"text_diff_{op_name}_{v2}_{v1}"
            sdom = domain.replace(".", "-")
            link = f"l-onnx-op{sdom}-{op_name.lower()}-d{v2}-{v1}"
            d_links[int(v2), int(v1)] = link
            content = "\n".join(
                [
                    "",
                    f".. _{link}:",
                    "",
                    title,
                    "=" * len(title),
                    "",
                    "Next section compares an older to a newer version of the same operator ",
                    "after both definition are converted into markdown text.",
                    "Green means an addition to the newer version, red means a deletion.",
                    "Anything else is unchanged.",
                    "",
                    ".. raw:: html",
                    "",
                    textwrap.indent(diff, "    "),
                ]
            )
            filename = os.path.join(folder, name + ".rst")
            if os.path.exists(filename):
                with open(filename, encoding="utf-8") as f:
                    old_content = f.read()
                    write = old_content != content
            else:
                write = True
            if write:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
            pieces.append(f"    {name}")

        pieces.extend(["", spl[i]])

    return "\n".join(pieces), d_links


def change_style(name: str) -> str:
    """
    Switches from *AaBb* into *aa_bb*.
    :param name: name to convert
    :return: converted name
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return s2 if not keyword.iskeyword(s2) else s2 + "_"


def _process_example(code: str) -> str:
    """
    Add necessary imports to make the example work.
    """
    code = code.replace("", "")
    missing_imports = ["import numpy as np", "import onnx"]
    elements = [*missing_imports, "", "", code.strip("\n"), ""]
    return "\n".join(elements)


def get_onnx_example(op_name):
    """
    Retrieves examples associated to one operator
    stored in onnx packages.
    :param op_name: operator name
    :param fmt: rendering format
    :return: dictionary
    """
    modules = [
        f"onnx.backend.test.case.node.{op_name.lower()}",
        f"onnx.backend.test.case.node.{change_style(op_name).lower()}",
    ]
    module = None
    for m in modules:
        try:
            mod = importlib.import_module(m)
            module = m
        except ImportError:
            continue
    if module is None:
        # Unable to find an example for 'op_name'.
        return {}
    results = {}
    for v in mod.__dict__.values():
        if not isinstance(v, _Exporter):
            continue
        code_cls = inspect.getsource(v)
        codes = code_cls.split("@staticmethod")
        for me in v.__dict__:
            if not me.startswith("export"):
                continue
            sub = f" {me}()"
            found = None
            for code in codes:
                if sub in code:
                    found = code
            if found is None:
                raise RuntimeError(f"Unable to find {sub!r} in\n{code_cls}")  # pragma: no cover
            found = textwrap.dedent(found)
            lines = found.split("\n")
            first = 0
            for i in range(len(lines)):  # pylint: disable=C0200
                if lines[i].startswith("def "):
                    first = i + 1
            found = textwrap.dedent("\n".join(lines[first:]))
            key = me[len("export") :]
            if not key:
                key = "default"
                if key in results:
                    key = f"example {len(results) + 1}"
            results[key] = _process_example(found)
    return results


def is_last_schema(sch: OpSchema) -> bool:
    """
    Tells if this is the most recent schema for this operator.
    :param sch: schema
    :return: True
    """
    try:
        last = get_schema(sch.name, domain=sch.domain)
    except SchemaError:  # pragma: no cover
        # raise RuntimeError(
        #     "Unable to find schema for operator %r and domain %r."
        #     "" % (sch.name, sch.domain))
        return True
    return last.since_version == sch.since_version


def onnx_documentation_folder(folder, title="ONNX Operators in onnxruntime", flog=None, max_opsets=None):
    """
    Creates documentation in a folder for all known
    ONNX operators defined in onnxruntime or a subset.
    :param folder: folder where to write the documentation
    :param title: index title
    :param flog: logging function
    :param max_opsets: included operator definition up to this opsets
    :return: list of creates files
    """

    class _Table:
        def __init__(self, ops, domain, title=None):
            self.title = title or domain
            self.domain = domain
            self.ops = ops

        @property
        def domain_name(self):
            title = self.domain
            if not title:
                title = "ai.onnx"
            return title

        def render(self, indent=""):
            table_dom = [""]
            table_dom.extend(
                [
                    ".. list-table::",
                    "    :widths: 10 10 10",
                    "    :header-rows: 1",
                    "",
                    "    * - operator",
                    "      - versions",
                    "      - differences",
                ]
            )

            for op in self.ops:
                name = op["name"]
                dom = self.domain.replace(".", "-")
                table_dom.append(f"    * - :ref:`l-onnx-doc{dom}-{name}`")
                versions = list(reversed(sorted((k, v) for k, v in op["links"].items() if isinstance(k, int))))
                col1 = ", ".join(f":ref:`{k} <{v}>`" for k, v in versions)
                diffs = list(reversed(sorted((k, v) for k, v in op["links"].items() if isinstance(k, tuple))))
                col2 = ", ".join(f":ref:`{k[1]}/{k[0]} <{v}>`" for k, v in diffs)
                table_dom.append(f"      - {col1}")
                table_dom.append(f"      - {col2}")
            table_dom.append("")
            if indent:
                for i in range(len(table_dom)):
                    table_dom[i] = indent + table_dom[i]
            res = "\n".join(table_dom)
            return res

    all_schemas_available = _get_all_schemas_with_history()

    # filter out operator under development
    all_schemas = {}
    for domain, ops in all_schemas_available.items():
        max_version = None if max_opsets is None else max_opsets.get(domain, None)
        d = {}
        for op, schemas in ops.items():
            vers = {}
            for version, schema in schemas.items():
                if max_version is not None and version > max_version:
                    continue
                vers[version] = schema
            d[op] = vers
        all_schemas[domain] = d

    if not os.path.exists(folder):
        os.makedirs(folder)

    pages = []
    tables = []

    # loop on domains
    for dom in sorted(all_schemas):
        sdom = dom if dom else "ai.onnx"
        dom_pages = []

        do = all_schemas[dom]
        if len(do) == 0:
            continue

        # loop on operators
        for op in sorted(do):
            if flog is not None:
                flog(f"generate page for onnx {dom!r} - {op!r}")  # pragma: no cover
            page_name = f"onnx_{dom.replace('.', '')}_{op}"
            doc, d_links = get_rst_doc(folder, op, domain=dom, version=None, example=True, diff=True)
            if not dom:
                main = op
            else:
                main = f"{dom} - {op}"
            sdom = dom.replace(".", "-")
            ref_link = f".. _l-onnx-doc{sdom}-{op}:"
            rows = [
                "",
                ref_link,
                "",
                "=" * len(main),
                main,
                "=" * len(main),
                "",
                doc,
            ]

            full = os.path.join(folder, page_name + ".rst")
            content = "\n".join(rows)
            if os.path.exists(full):
                with open(full, encoding="utf-8") as f:
                    old_content = f.read()
                write = old_content != content
            else:
                write = True
            if write:
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
            pages.append(full)
            dom_pages.append({"name": op, "links": d_links})

        tables.append(_Table(dom_pages, dom, sdom))

    # final
    tmpl = _template_main
    index = tmpl.render(pages=pages, tabs=tables, os=os, len=len, title=title)
    index = _clean_unicode(index)
    page_name = os.path.join(folder, "index.rst")
    with open(page_name, "w", encoding="utf-8") as f:
        f.write(index)
    pages.append(page_name)
    return pages


def _generate_op_doc(app):
    logger = logging.getLogger(__name__)
    folder = app.config.onnx_doc_folder
    max_opsets = app.config.max_opsets
    onnx_documentation_folder(folder, flog=logger.info, max_opsets=max_opsets)


def setup(app):
    """
    Sphinx extension `onnx_sphinx` displays documentation
    of all ONNX Operators.
    """
    import sphinx

    app.add_config_value("onnx_doc_folder", "operators", "env")
    app.add_config_value("max_opsets", {}, "env")
    app.connect("builder-inited", _generate_op_doc)
    return {"version": sphinx.__display_version__, "parallel_read_safe": True}


if "debug" in sys.argv:
    print("DEBUG")
    onnx_documentation_folder("_debug", flog=print)
    print("END")
