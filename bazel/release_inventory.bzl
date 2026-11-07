# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Hash actual language-release outputs without implying an unbuilt platform matrix."""

def _release_inventory_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".json")
    args = ctx.actions.args()
    args.add("--output", output)
    args.add("--version-file", ctx.file.version)
    args.add("--platform", ctx.attr.platform)
    args.add("--variant", ctx.attr.variant)
    inputs = [ctx.file.version]
    for target in ctx.attr.packages:
        for file in target[DefaultInfo].files.to_list():
            args.add("--artifact", str(target.label) + "=" + file.path)
            inputs.append(file)
    ctx.actions.run(
        executable = ctx.executable.tool,
        tools = [ctx.attr.tool[DefaultInfo].files_to_run],
        arguments = [args],
        inputs = inputs,
        outputs = [output],
        mnemonic = "OnnxRuntimeReleaseInventory",
    )
    return [DefaultInfo(files = depset([output]))]

release_inventory = rule(
    implementation = _release_inventory_impl,
    attrs = {
        "packages": attr.label_list(mandatory = True, allow_files = True),
        "version": attr.label(default = "//:VERSION_NUMBER", allow_single_file = True),
        "platform": attr.string(mandatory = True),
        "variant": attr.string(default = "cpu"),
        "tool": attr.label(
            default = "//:release_inventory_tool",
            executable = True,
            cfg = "exec",
        ),
    },
)
