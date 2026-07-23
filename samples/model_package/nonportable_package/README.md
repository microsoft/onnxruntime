# Non-portable (installed) package

An **installed**-layout model package whose model files live **outside** the package
directory. The package references them by **absolute** path.

This is useful when large model assets are managed separately from the (small)
package metadata — e.g. a shared on-disk cache, a system install location, or a
download folder that several packages point at.

## Layout

```
samples/model_package/
├── nonportable_package/                 # <- this package (metadata only)
│   ├── manifest.json                    #    "layout": "installed"
│   ├── prefill/
│   │   ├── cpu/ort_info.template.json
│   │   └── ov/ort_info.template.json
│   ├── iter/
│   │   ├── cpu/ort_info.template.json
│   │   └── ov/ort_info.template.json
│   ├── resolve.py                       #    fills absolute paths -> ort_info.json
│   └── README.md
└── external_assets/                     # <- model files, OUTSIDE the package
    ├── base/                            #    base CPU models + shared weights
    │   ├── prefill.onnx
    │   ├── iter.onnx
    │   └── weights.data
    └── compiled/                        #    compiled OpenVINO models + shared bin
        ├── prefill.ctx.onnx
        ├── iter.ctx.onnx
        └── prefill.ctx_OpenVINOExecutionProvider.bin
```

## Why a resolve step?

The `installed` layout allows absolute paths and `..` segments — but neither is a good
fit for something committed to a repository:

- **Absolute paths** are machine-specific; they can't be committed.
- **`..` relative segments** (e.g. `../external_assets/base/prefill.onnx`) *are*
  supported, but they are fragile (they break if the package is moved to a different
  depth) and easy to get wrong.

So each variant ships an `ort_info.template.json` containing the placeholder token
`__ASSETS_DIR__`, and [`resolve.py`](resolve.py) rewrites the templates into real
`ort_info.json` files with the absolute path to `external_assets/` on **this** machine.

Template (`prefill/ov/ort_info.template.json`):

```json
{
  "model_file": "__ASSETS_DIR__/compiled/prefill.ctx.onnx",
  "session_options": {
    "ep.share_ep_contexts": "1",
    "session.model_external_initializers_file_folder_path": "__ASSETS_DIR__/base",
    "ep.context_file_path": "__ASSETS_DIR__/compiled/prefill.ctx.onnx"
  }
}
```

After `resolve.py`, `__ASSETS_DIR__` becomes the absolute path to `external_assets/`,
and the base/`weights.data` and compiled/`.bin` are read from those external folders.
The wiring is otherwise identical to the portable package (see its README for what each
option does).

## Use it

```powershell
# 1) Resolve external paths (run once per checkout / after moving the tree).
python resolve.py
#    or point at a custom assets location:
python resolve.py --assets D:\models\tiny-mlp\external_assets

# 2) Load the package.
..\cpp\model_package_sample.exe <onnxruntime.dll> . <openvino_plugin.dll>
```

Before `resolve.py` runs, the `ort_info.json` files do not exist and loading the
package fails with a clear "does not exist" error — that's expected. Re-run
`resolve.py` whenever you move the checkout or the external assets.

> `ort_info.json` (the resolved output) is machine-specific and is intended to be
> git-ignored; commit only the `*.template.json` files.
