# Quick Overview
- transforms.py - graph transform implementations that starts with transform_* 

- transform_model.py - calls transforms.py to transform <original_model> to <transformed_model>

Example
`python transform.py --original_model <original_model> --transformed_model <transformed_model> --transform_sequence <list of transform sequence(function names in transforms.py)>`

- transform_pipeline.py - calls transform_model.py and compare_model_outputs.py for all models

Example
`python .\transform_pipeline.py --config_file <model_config.json>`
