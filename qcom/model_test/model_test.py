#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

import argparse
import logging
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, cast, get_args

import jsonc
import numpy as np
import onnx

import onnxruntime

DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-5


BackendT = Literal["cpu", "gpu", "htp"]


class ModelTestDef(NamedTuple):
    model_root: Path
    backend_type: BackendT
    rtol: float
    atol: float
    enable_context: bool
    enable_cpu_fallback: bool

    def __repr__(self) -> str:
        return (
            f"{self.model_root.name}|{self.backend_type}|rtol:{self.rtol}|atol:{self.atol}"
            f"|enable_context:{self.enable_context}|cpu_fallback:{self.enable_cpu_fallback}"
        )


class ModelTestCase:
    def __init__(self, model_def: ModelTestDef) -> None:
        self.__model_root = model_def.model_root
        self.__rtol = model_def.rtol
        self.__atol = model_def.atol

        session_options = onnxruntime.SessionOptions()

        if not model_def.enable_cpu_fallback and model_def.backend_type != "cpu":
            session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        if model_def.enable_context:
            context_model_path = model_def.model_root / f"{model_def.model_root.name}_ctx.onnx"
            if context_model_path.exists():
                logging.debug(f"Clobbering stale context in {context_model_path}")
                context_model_path.unlink()
            session_options.add_session_config_entry("ep.context_enable", "1")
            session_options.add_session_config_entry("ep.context_file_path", str(context_model_path))

        self.__session = onnxruntime.InferenceSession(
            model_def.model_root / f"{model_def.model_root.name}.onnx",
            sess_options=session_options,
            providers=["QNNExecutionProvider"],
            provider_options=[{"backend_type": model_def.backend_type}],
        )

    @property
    def inputs(self) -> list[dict[str, np.ndarray]]:
        return [dict(zip(self.input_names, ip, strict=False)) for ip in self.input_protos]

    @property
    def input_names(self) -> list[str]:
        return [n.name for n in self.__session.get_inputs()]

    @property
    def input_paths(self) -> list[list[Path]]:
        return [list(ds.glob("input_*.pb")) for ds in self.__model_root.glob("test_data_set_*")]

    @property
    def input_protos(self) -> list[list[np.ndarray]]:
        return self.__tensors_from_files(self.input_paths)

    @property
    def output_paths(self) -> list[list[Path]]:
        return [list(ds.glob("output_*.pb")) for ds in self.__model_root.glob("test_data_set_*")]

    @property
    def output_protos(self) -> list[list[np.ndarray]]:
        return self.__tensors_from_files(self.output_paths)

    @staticmethod
    def __tensors_from_files(data_sets: Iterable[Iterable[Path]]) -> list[list[np.ndarray]]:
        return [
            [onnx.numpy_helper.to_array(onnx.TensorProto.FromString(f.read_bytes())) for f in ds] for ds in data_sets
        ]

    def run(self) -> None:
        assert len(self.inputs) == len(self.output_protos)

        for ds_idx in range(len(self.inputs)):
            actual = cast(Sequence[np.ndarray], self.__session.run([], self.inputs[ds_idx]))
            expected = cast(Sequence[np.ndarray], self.output_protos[ds_idx])

            assert len(actual) == len(expected)
            for i in range(len(actual)):
                np.testing.assert_allclose(actual[i], expected[i], atol=self.__atol, rtol=self.__rtol)


class ModelTestSuite:
    def __init__(
        self,
        suite_root: Path,
        backend_type: BackendT,
        rtol: float | None,
        atol: float | None,
        enable_context: bool,
        enable_cpu_fallback: bool,
    ) -> None:
        self.__suite_root = suite_root
        self.__backend_type: BackendT = backend_type
        config = self.__parse_config()
        self.__enable_context = enable_context
        self.__enable_cpu_fallback = enable_cpu_fallback
        self.__default_rtol = rtol if rtol else cast(float, config.get("rtol_default", DEFAULT_RTOL))
        self.__default_atol = atol if atol else cast(float, config.get("atol_default", DEFAULT_ATOL))
        self.__rtol_overrides = cast(dict[str, float], config.get("rtol_overrides", {}))
        self.__atol_overrides = cast(dict[str, float], config.get("atol_overrides", {}))

    def __parse_config(self) -> dict[str, float | dict[str, float]]:
        # I don't love that this file sits in the directory above the test suite, but
        # that's that the ORT tests do so we're sticking with it.
        config_path = self.__suite_root.parent / "onnx_backend_test_series_overrides.jsonc"
        if not config_path.exists():
            return {}
        return jsonc.load(config_path.open("rt"))

    @property
    def tests(self) -> Generator[ModelTestDef, None, None]:
        for test_root in self.__suite_root.iterdir():
            model_name = test_root.name
            rtol = self.__rtol_overrides.get(model_name, self.__default_rtol)
            atol = self.__atol_overrides.get(model_name, self.__default_atol)
            yield ModelTestDef(
                test_root, self.__backend_type, rtol, atol, self.__enable_context, self.__enable_cpu_fallback
            )

    def run(self) -> None:
        for test in self.tests:
            ModelTestCase(test).run()


def initialize_logging(log_name: str) -> None:
    log_format = f"[%(asctime)s] [{log_name}] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format, force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", default="htp", choices=get_args(BackendT), help="QNN backend to use.")
    parser.add_argument("--rtol", type=float, help="Relative tolerance")
    parser.add_argument("--atol", type=float, help="Absolute tolerance")
    parser.add_argument("--enable-context", action="store_true", help="[HTP only] Create a context cache.")
    parser.add_argument("--enable-cpu-fallback", action="store_true", help="Allow execution to fall back to CPU.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path, metavar="MODEL_DIR", help="Path to a single model's directory.")
    model_group.add_argument("--suite", type=Path, metavar="SUITE_DIR", help="Path to a test suite directory.")

    args = parser.parse_args()

    initialize_logging("model_test.py")

    if args.model:
        rtol = args.rtol if args.rtol else DEFAULT_RTOL
        atol = args.atol if args.atol else DEFAULT_ATOL
        ModelTestCase(
            ModelTestDef(
                args.model,
                args.backend,
                rtol=rtol,
                atol=atol,
                enable_context=args.enable_context,
                enable_cpu_fallback=args.enable_cpu_fallback,
            )
        ).run()
    elif args.suite:
        ModelTestSuite(
            args.suite,
            args.backend,
            rtol=args.rtol,
            atol=args.atol,
            enable_context=args.enable_context,
            enable_cpu_fallback=args.enable_cpu_fallback,
        ).run()
    else:
        raise RuntimeError("Unknown test mode")
