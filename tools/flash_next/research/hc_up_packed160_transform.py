from build_qmoe_node_replay import replace_once
import hc_up_packed80_transform as packed80


FLAG = "ORT_WEBGPU_HC_UP_PACKED160"
NAME = "HCUpPreMixPacked160F16"


def selected_program(source):
    program = packed80.selected_program(source)
    program = program.replace("HCUpPreMixPacked80Program", "HCUpPreMixPacked160Program")
    for original, replacement in (
            (packed80.NAME, NAME),
            ("partials: array<f16, 264>", "partials: array<f16, 528>"),
            ("gates: array<f16, 8>", "gates: array<f16, 16>"),
            ("workgroup_idx * 2u + column", "workgroup_idx * 4u + column")):
        program = replace_once(program, original, replacement)
    return program


def transform(source):
    actual = packed80.transform(source)
    actual = replace_once(actual, packed80.selected_program(source), selected_program(source))
    actual = actual.replace("HCUpPreMixPacked80Program", "HCUpPreMixPacked160Program")
    actual = actual.replace(packed80.FLAG, FLAG).replace("packed80", "packed160")
    actual = replace_once(actual, "SetWorkgroupSize(selected ? 80 : 128)",
                          "SetWorkgroupSize(selected ? 160 : 128)")
    return replace_once(actual, "SetDispatchGroupSize(parallel2 == 1 ? 1280 : 2560)",
                         "SetDispatchGroupSize(selected ? 640 : (parallel2 == 1 ? 1280 : 2560))")
