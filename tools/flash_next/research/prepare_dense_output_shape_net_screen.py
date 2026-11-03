from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def transform(source):
    source = source.replace("run_dense_output_shape_model", "run_dense_output_shape_net_model")
    source = replace_once(source, "    args = parser.parse_args()", '''    args = parser.parse_args()
    if args.repeat or args.context != 2048:
        parser.error("Only initial 2K original-provider net comparison is supported")''')
    source = replace_once(source, "    provider = None", "    providers = {}")
    source = replace_once(source,
        '            provider = proof["provider_sha256"] if provider is None else provider\n'
        '            assert proof["provider_sha256"] == raw["plugin_sha256"] == provider',
        '            providers.setdefault(mode, proof["provider_sha256"])\n'
        '            assert proof["provider_sha256"] == raw["plugin_sha256"] == providers[mode]\n'
        '            assert proof["original_provider_control"] == (mode == "control")')
    source = replace_once(source, '    manifest["status"] = "completed"',
        '''    assert providers["control"] != providers["shape"]
    assert providers["control"] == "0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4"
    provider = providers["shape"]
    manifest["status"] = "completed"''')
    source = replace_once(source, 'context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,',
        'context=args.context, repeat=args.repeat, provider_sha256=provider, rows=rows,\n'
        '                  original_baseline_provider_sha256=providers["control"],')
    return source


def main():
    target = ROOT / "run_dense_output_shape_net_screen.py"
    if target.exists():
        raise FileExistsError(target)
    with target.open("x") as stream:
        stream.write(transform((ROOT / "run_dense_output_shape_screen.py").read_text()))
    print(target.name)


if __name__ == "__main__":
    main()
