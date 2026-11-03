import json
from pathlib import Path
import sys

import onnxruntime_genai as og


root = Path(sys.argv[1])
for name in ("shared", "growing", "fixed_missing_length"):
    model = og.Model(str(root / name))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=40, min_length=40, do_sample=False)
    try:
        generator = og.Generator(model, params)
    except RuntimeError as error:
        if name != "fixed_missing_length" or str(error) != (
            "IndexerCache: shared caches require an int32 past_sequence_length input"
        ):
            raise
        print(json.dumps({"mode": name, "expected_contract_error": str(error)}), flush=True)
    else:
        print(json.dumps({"mode": name, "generator_construction": "passed"}), flush=True)
        del generator
    del params, model
