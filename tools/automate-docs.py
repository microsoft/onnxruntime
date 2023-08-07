import requests

# Note - generation of different markdown files cannot be modularized in different Python methods as indendation matters.

with open('./docs/extensions/index.md', 'w') as f:
    index_link = "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/README.md"
    docs = requests.get(index_link)
    intro = """---
title: Extensions
has_children: true
nav_order: 7
---

"""
    img = """<img src="../../images/combine-ai-extensions-img.png" alt="Pre and post-processing custom operators for vision, text, and NLP models" width="100%"/>
<sub>This image was created using <a href="https://github.com/sayanshaw24/combine" target="_blank">Combine.AI</a>, which is powered by Bing Chat, Bing Image Creator, and EdgeGPT.</sub>

"""
    md = intro + docs.text[:docs.text.index("## Quickstart")] + img + docs.text[docs.text.index("## Quickstart"):docs.text.index("(LICENSE)")] + "(https://github.com/microsoft/onnxruntime-extensions/blob/main/LICENSE)"
    f.write(md)

with open('./docs/extensions/development.md', 'w') as f:
    development_link = "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/docs/development.md"
    ci_matrix_link = "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/docs/ci_matrix.md"
    docs = requests.get(development_link)
    ci_matrix = requests.get(ci_matrix_link)
    intro = """---
title: Development
description: Instructions for building and developing ORT Extensions.
parent: Extensions
nav_order: 2
---

"""
    custom_build_intro = "For instructions on building ONNX Runtime with onnxruntime-extensions for Java package, see [here](./custom-build.md)\n\nRun "
    md = intro + docs.text[:docs.text.index("(<./ci_matrix.md>)")] + "(./development.md#dependencies)\n" + docs.text[docs.text.index("(<./ci_matrix.md>)")+19:docs.text.index("## Java package")+16] + custom_build_intro + docs.text[docs.text.index("`bash ./build.sh -DOCOS_BUILD_JAVA=ON`"):] + "\n## Dependencies\n" + ci_matrix.text[ci_matrix.text.index("The matrix"):]
    f.write(md)

with open('./docs/extensions/custom-build.md', 'w') as f:
    custom_build_link = "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/docs/custom_build.md"
    docs = requests.get(custom_build_link)
    intro = """---
title: Custom Build
description: Instructions for building ONNX Runtime with onnxruntime-extensions for Java package.
parent: Development
nav_order: 2
---

"""
    md = intro + docs.text
    f.write(md)

with open('./docs/extensions/pyop.md', 'w') as f:
    pyop_link = "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/docs/pyop.md"
    docs = requests.get(pyop_link)
    intro = """---
title: Python Operator
description: Instructions to create a custom operator using Python functions and ORT inference integration.
parent: Extensions
nav_order: 4
---

"""
    md = intro + "# Creating custom operators using Python functions\n\n" + docs.text[docs.text.index("Custom operators"):docs.text.index("Here is an example")] + "Here is an example: [test_pyops.py](https://github.com/microsoft/onnxruntime-extensions/blob/main/test/test_pyops.py)"
    f.write(md)
