# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import aie.utils as aie_utils
import ml_dtypes
import numpy as np
import pyxrt
from aie import ir
from aie.dialects import aie, aiex, memref
from aie.extras.context import mlir_mod_ctx
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from iron.common import compilation as comp
from iron.common.base import AIEOperatorBase, MLIROperator
from iron.common.utils import XRTSubBuffer


def _extract_runtime_sequence_arg_types(dev_op: Any) -> list[Any]:
    for nested_op in dev_op.body_region.blocks[0].operations:
        if nested_op.operation.name == "aie.runtime_sequence":
            if hasattr(nested_op, "body") and hasattr(nested_op.body, "blocks"):
                if nested_op.body.blocks:
                    entry_block = nested_op.body.blocks[0]
                    return [
                        entry_block.arguments[i].type
                        for i in range(len(entry_block.arguments))
                    ]
    raise RuntimeError("Could not find runtime sequence in device operation")


def _get_child_mlir_module(mlir_artifact: comp.PythonGeneratedMLIRArtifact) -> Any:
    gen = mlir_artifact.generator
    spec = __import__("importlib.util").util.spec_from_file_location(
        gen.source_path.name, gen.source_path
    )
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callback = getattr(module, gen.fn_name)
    return callback(*gen.args, **gen.kwargs)


def _write_fused_mlir(
    *,
    filename: str,
    operator_mlir_map: dict[str, comp.PythonGeneratedMLIRArtifact],
    runlist: list[tuple[str, ...]],
    subbuffer_layout: dict[str, tuple[str, int, int]],
    buffer_sizes: dict[str, int],
    buffer_order: list[str],
    slice_info: dict[str, tuple[str, int, int]],
) -> None:
    device_mlir_strings = {}
    device_ty = None
    sequence_arg_types = {}
    for op_name, mlir_artifact in operator_mlir_map.items():
        mlir_module = _get_child_mlir_module(mlir_artifact)
        device_ops = [
            op for op in mlir_module.body.operations if isinstance(op, aie.DeviceOp)
        ]
        if len(device_ops) != 1:
            raise ValueError(
                f"Expected exactly one device operation for {op_name}, "
                f"got {len(device_ops)}"
            )
        device_op = device_ops[0]
        if device_ty is None:
            device_ty = device_op.device
        device_mlir_strings[op_name] = str(device_op)
        sequence_arg_types[op_name] = _extract_runtime_sequence_arg_types(device_op)

    with mlir_mod_ctx() as ctx:
        for op_name, device_str in device_mlir_strings.items():
            dev_op = aie.DeviceOp.parse(device_str)
            dev_op.sym_name = ir.StringAttr.get(op_name)
            ctx.module.body.append(dev_op)

        @aie.device(device_ty)
        def main():
            buf_dtype = np.dtype[ml_dtypes.bfloat16]
            itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
            sequence_types = [
                np.ndarray[(buffer_sizes[name] // itemsize,), buf_dtype]
                for name in buffer_order
            ]

            def emit_sequence(runtime_buffers):
                consolidated_buffers = dict(zip(buffer_order, runtime_buffers))
                configure_op = None
                last_op_name = None

                for op_name, *buffer_names in runlist:
                    expected_arg_types = sequence_arg_types[op_name]
                    if configure_op is None or op_name != last_op_name:
                        configure_op = aiex.ConfigureOp(
                            ir.FlatSymbolRefAttr.get(op_name)
                        )
                        configure_body = configure_op.body.blocks.append()
                        last_op_name = op_name

                    with ir.InsertionPoint(configure_body):
                        buffer_ssa_values = []
                        for idx, buf_name in enumerate(buffer_names):
                            if buf_name in slice_info:
                                base_name, start, end = slice_info[buf_name]
                                buf_type, parent_offset, _parent_length = (
                                    subbuffer_layout[base_name]
                                )
                                offset = parent_offset + start
                                length = end - start
                            else:
                                buf_type, offset, length = subbuffer_layout[buf_name]

                            consolidated_buf = consolidated_buffers[buf_type]
                            offset_elements = offset // itemsize
                            size_elements = length // itemsize
                            subview = memref.subview(
                                consolidated_buf,
                                [offset_elements],
                                [size_elements],
                                [1],
                            )

                            target_type = expected_arg_types[idx]
                            expected_memref = ir.MemRefType(target_type)
                            target_shape = [
                                expected_memref.shape[i]
                                for i in range(expected_memref.rank)
                            ]
                            expected_size = np.prod(target_shape)
                            assert expected_size == size_elements, (
                                f"Size mismatch for buffer {buf_name!r}: "
                                f"MLIR expected {expected_size}, layout has "
                                f"{size_elements}"
                            )
                            strides = []
                            stride = 1
                            for dim in reversed(target_shape):
                                strides.insert(0, stride)
                                stride *= dim
                            result_type = ir.MemRefType.get(
                                target_shape, ir.BF16Type.get()
                            )
                            reinterpreted = memref.reinterpret_cast(
                                result=result_type,
                                source=subview,
                                offsets=[],
                                sizes=[],
                                strides=[],
                                static_offsets=[0],
                                static_sizes=target_shape,
                                static_strides=strides,
                            )
                            buffer_ssa_values.append(reinterpreted)

                        aiex.RunOp(
                            ir.FlatSymbolRefAttr.get("sequence"),
                            buffer_ssa_values,
                        )

            if len(sequence_types) == 3:

                @aiex.runtime_sequence(*sequence_types)
                def sequence(input_buf, output_buf, scratch_buf):
                    emit_sequence((input_buf, output_buf, scratch_buf))

            elif len(sequence_types) == 4:

                @aiex.runtime_sequence(*sequence_types)
                def sequence(input_buf, output_buf, scratch_buf, external0):
                    emit_sequence((input_buf, output_buf, scratch_buf, external0))

            elif len(sequence_types) == 5:

                @aiex.runtime_sequence(*sequence_types)
                def sequence(input_buf, output_buf, scratch_buf, external0, external1):
                    emit_sequence(
                        (input_buf, output_buf, scratch_buf, external0, external1)
                    )

            elif len(sequence_types) == 6:

                @aiex.runtime_sequence(*sequence_types)
                def sequence(
                    input_buf,
                    output_buf,
                    scratch_buf,
                    external0,
                    external1,
                    external2,
                ):
                    emit_sequence(
                        (
                            input_buf,
                            output_buf,
                            scratch_buf,
                            external0,
                            external1,
                            external2,
                        )
                    )

            else:
                raise ValueError(
                    f"Unsupported fused runtime buffer count: {len(sequence_types)}"
                )

    path = Path(filename)
    contents = str(ctx.module)
    if path.exists() and path.read_text() == contents:
        return
    path.write_text(contents)


def _file_stamp(path: str | Path) -> tuple[int | None, int | None]:
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None, None
    return stat.st_mtime_ns, stat.st_size


def _source_stamps(artifact: comp.CompilationArtifact) -> list[dict[str, Any]]:
    stamps = []
    for dep in artifact.dependencies.bfs():
        if isinstance(dep, comp.SourceArtifact):
            stamps.append(
                {
                    "filename": str(dep.filename),
                    "stamp": _file_stamp(dep.filename),
                }
            )
    return stamps


def _fused_mlir_signature(
    *,
    operator_mlir_map: dict[str, comp.PythonGeneratedMLIRArtifact],
    runlist: list[tuple[str, ...]],
    subbuffer_layout: dict[str, tuple[str, int, int]],
    buffer_sizes: dict[str, int],
    buffer_order: list[str],
    slice_info: dict[str, tuple[str, int, int]],
    kernel_objects: list[comp.KernelObjectArtifact],
) -> str:
    payload = {
        "fusion_source": _file_stamp(__file__),
        "operators": {
            op_name: {
                "filename": mlir_artifact.filename,
                "generator_source": str(mlir_artifact.generator.source_path),
                "generator_source_stamp": _file_stamp(
                    mlir_artifact.generator.source_path
                ),
                "fn_name": mlir_artifact.generator.fn_name,
                "args": repr(mlir_artifact.generator.args),
                "kwargs": repr(mlir_artifact.generator.kwargs),
            }
            for op_name, mlir_artifact in sorted(operator_mlir_map.items())
        },
        "runlist": runlist,
        "subbuffer_layout": subbuffer_layout,
        "buffer_sizes": buffer_sizes,
        "buffer_order": buffer_order,
        "slice_info": slice_info,
        "kernel_objects": [
            {
                "filename": obj.filename,
                "extra_flags": obj.extra_flags,
                "rename_symbols": obj.rename_symbols,
                "prefix_symbols": obj.prefix_symbols,
                "sources": _source_stamps(obj),
            }
            for obj in kernel_objects
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fused_mlir_cache_valid(mlir_path: Path, signature_path: Path, signature: str):
    if not mlir_path.exists() or not signature_path.exists():
        return False
    return signature_path.read_text().strip() == signature


class FusedMLIROperator(AIEOperatorBase):
    """Local fused operator with optional external runtime buffers.

    Buffers named in ``input_args`` are packed into ``input``; buffers named in
    ``output_args`` are packed into ``output``; buffers named in
    ``external_args`` are packed into their named external buffer type. All
    remaining buffers are packed into ``scratch``.
    """

    def __init__(
        self,
        name,
        runlist,
        input_args,
        output_args,
        buffer_sizes=None,
        external_args=None,
        compile_mode="full_elf",
        *args,
        **kwargs,
    ):
        if not all(
            isinstance(op, MLIROperator) and all(isinstance(buf, str) for buf in bufs)
            for op, *bufs in runlist
        ):
            raise TypeError(
                "runlist entries must be (MLIROperator, *str) tuples; "
                "each operator must be an MLIROperator and each buffer name must be a str"
            )
        super().__init__(*args, **kwargs)
        self.name = name
        self.runlist = runlist
        self.input_args = list(input_args)
        self.output_args = list(output_args)
        self.explicit_buffer_sizes = buffer_sizes or {}
        self.external_args = {
            name: list(args) for name, args in (external_args or {}).items()
        }
        self.compile_mode = compile_mode
        if self.compile_mode not in {"full_elf", "full_elf_dynamic", "xclbin"}:
            raise ValueError(f"unsupported fused compile mode: {self.compile_mode}")
        reserved = {"input", "output", "scratch"}
        if reserved.intersection(self.external_args):
            raise ValueError("external_args cannot use input/output/scratch names")

    @property
    def buffer_order(self):
        return ["input", "output", "scratch", *self.external_args.keys()]

    def get_kernel_artifacts(self):
        kernel_artifacts = []
        seen: dict[int, object] = {}
        unique_operators = [
            seen.setdefault(id(op), op) for op, *_ in self.runlist if id(op) not in seen
        ]
        for idx, op in enumerate(unique_operators):
            objs = op.get_kernel_artifacts()
            for obj in objs:
                if self.compile_mode == "full_elf":
                    obj.filename = f"op{idx}_{obj.filename}"
                    obj.prefix_symbols = f"op{idx}_"
            kernel_artifacts.extend(objs)
        return kernel_artifacts

    def _operator_mlir_map_and_runlist(self):
        operator_mlir_map = {}
        comp_runlist = []
        op_names = {}
        seen: dict[int, object] = {}
        unique_operators = [
            seen.setdefault(id(op), op) for op, *_ in self.runlist if id(op) not in seen
        ]
        for idx, op in enumerate(unique_operators):
            mlir_artifact = op.get_mlir_artifact()
            if self.compile_mode == "full_elf" and op.get_kernel_artifacts():
                mlir_artifact.generator.kwargs["func_prefix"] = f"op{idx}_"
            op_name = f"op{idx}_{op.__class__.__name__}"
            op_names[id(op)] = op_name
            operator_mlir_map[op_name] = mlir_artifact

        for op, *bufs in self.runlist:
            comp_runlist.append((op_names[id(op)], *bufs))
        return operator_mlir_map, comp_runlist

    def _calculate_buffer_layout(self):
        args = {}
        sliced_buffers = {}

        for op, *bufs in self.runlist:
            arg_specs = op.get_arg_spec()
            if len(arg_specs) != len(bufs):
                raise ValueError(
                    f"Number of buffers ({len(bufs)}) must match operator "
                    f"argument specification ({len(arg_specs)}) for {op!r}"
                )
            for i, buf_name in enumerate(bufs):
                arg_spec = arg_specs[i]
                if "[" in buf_name and buf_name.endswith("]"):
                    base_name = buf_name[: buf_name.index("[")]
                    slice_part = buf_name[buf_name.index("[") + 1 : -1]
                    start, end = map(int, slice_part.split(":"))
                    sliced_buffers[buf_name] = (base_name, start, end, arg_spec)
                    if (
                        base_name not in args
                        and base_name not in self.explicit_buffer_sizes
                    ):
                        raise ValueError(
                            f"Sliced buffer {buf_name!r} requires explicit size for "
                            f"base buffer {base_name!r}"
                        )
                elif buf_name not in args:
                    args[buf_name] = arg_spec
                elif np.prod(args[buf_name].shape) != np.prod(arg_spec.shape):
                    raise ValueError(
                        f"Buffer {buf_name!r} has conflicting sizes: "
                        f"{args[buf_name].shape} vs {arg_spec.shape}"
                    )

        all_buffer_names = set(args) | set(sliced_buffers)
        for arg in self.input_args:
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Input argument {arg!r} not found in runlist buffers")
        for arg in self.output_args:
            if arg not in all_buffer_names and arg not in self.explicit_buffer_sizes:
                raise ValueError(f"Output argument {arg!r} not found in runlist buffers")
        for buffer_type, ext_args in self.external_args.items():
            for arg in ext_args:
                if (
                    arg not in all_buffer_names
                    and arg not in self.explicit_buffer_sizes
                ):
                    raise ValueError(
                        f"External argument {arg!r} for {buffer_type!r} not found"
                    )

        subbuffer_layout = {}
        slice_info = {}

        def add_buffers(buffer_type, args_list):
            offset = 0
            for arg in args_list:
                if arg in self.explicit_buffer_sizes:
                    length = self.explicit_buffer_sizes[arg]
                elif arg in args:
                    spec = args[arg]
                    length = int(np.prod(spec.shape) * np.dtype(spec.dtype).itemsize)
                else:
                    continue
                subbuffer_layout[arg] = (buffer_type, offset, length)
                offset += length
            return offset

        for buf_name, (base_name, start, end, _arg_spec) in sliced_buffers.items():
            slice_info[buf_name] = (base_name, start, end)

        assigned = set(self.input_args) | set(self.output_args)
        for ext_args in self.external_args.values():
            assigned.update(ext_args)

        buffer_sizes = {
            "input": add_buffers("input", self.input_args),
            "output": add_buffers("output", self.output_args),
        }
        for buffer_type, ext_args in self.external_args.items():
            buffer_sizes[buffer_type] = add_buffers(buffer_type, ext_args)

        scratch_args = [arg for arg in args if arg not in assigned]
        for explicit_buf in self.explicit_buffer_sizes:
            if explicit_buf not in assigned and explicit_buf not in scratch_args:
                scratch_args.append(explicit_buf)
        buffer_sizes["scratch"] = add_buffers("scratch", scratch_args)

        ordered_sizes = {name: buffer_sizes[name] for name in self.buffer_order}
        return subbuffer_layout, ordered_sizes, slice_info

    def set_up_artifacts(self):
        self.subbuffer_layout, self.buffer_sizes, self.slice_info = (
            self._calculate_buffer_layout()
        )
        operator_mlir_map, comp_runlist = self._operator_mlir_map_and_runlist()
        self.context.build_dir.mkdir(parents=True, exist_ok=True)
        mlir_path = self.context.build_dir / f"{self.name}_fused.mlir"
        kernel_objects = self.get_kernel_artifacts()
        signature_path = mlir_path.with_suffix(mlir_path.suffix + ".sig")
        signature = _fused_mlir_signature(
            operator_mlir_map=operator_mlir_map,
            runlist=comp_runlist,
            subbuffer_layout=self.subbuffer_layout,
            buffer_sizes=self.buffer_sizes,
            buffer_order=self.buffer_order,
            slice_info=self.slice_info,
            kernel_objects=kernel_objects,
        )
        if not _fused_mlir_cache_valid(mlir_path, signature_path, signature):
            _write_fused_mlir(
                filename=str(mlir_path),
                operator_mlir_map=operator_mlir_map,
                runlist=comp_runlist,
                subbuffer_layout=self.subbuffer_layout,
                buffer_sizes=self.buffer_sizes,
                buffer_order=self.buffer_order,
                slice_info=self.slice_info,
            )
            signature_path.write_text(signature + "\n")
        mlir_artifact = comp.SourceArtifact(mlir_path, available=True)
        if self.compile_mode in {"full_elf", "full_elf_dynamic"}:
            full_elf_artifact = comp.FullElfArtifact(
                f"{self.name}.elf",
                mlir_input=mlir_artifact,
                dependencies=[mlir_artifact] + kernel_objects,
            )
            self.add_artifacts([full_elf_artifact])
        else:
            self.xclbin_artifact = comp.XclbinArtifact(
                f"{self.name}.xclbin",
                mlir_input=mlir_artifact,
                dependencies=[mlir_artifact] + kernel_objects,
            )
            self.insts_artifact = comp.InstsBinArtifact(
                f"{self.name}.bin",
                mlir_input=mlir_artifact,
                dependencies=[mlir_artifact],
            )
            self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def compile(self, dry_run: bool = False):
        if not self.artifacts:
            self.set_up_artifacts()

        rules = self.context.compilation_rules
        if self.compile_mode == "full_elf_dynamic":
            rules = [
                DynamicFullElfCompilationRule(rule)
                if isinstance(rule, comp.AieccFullElfCompilationRule)
                else rule
                for rule in rules
            ]
        comp.compile(
            rules,
            self.artifacts,
            self.context.build_dir,
            dry_run=dry_run,
        )
        return self

    def get_arg_spec(self):
        raise NotImplementedError(
            "FusedMLIROperator does not expose a unified arg spec; use "
            "get_layout_for_buffer() to inspect individual buffers"
        )

    def get_callable(self):
        if self.compile_mode in {"full_elf", "full_elf_dynamic"}:
            return FusedFullELFCallable(self)
        return FusedXclbinCallable(self)

    def get_layout_for_buffer(self, buffer_name):
        if buffer_name in self.slice_info:
            buf_name, start, end = self.slice_info[buffer_name]
            buf_type, parent_start, _parent_length = self.get_layout_for_buffer(
                buf_name
            )
            return buf_type, parent_start + start, end - start
        return self.subbuffer_layout[buffer_name]


def load_elf(op):
    assert isinstance(op.artifacts[0], comp.FullElfArtifact)
    with open(op.artifacts[0].filename, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint32)


class DynamicFullElfCompilationRule(comp.CompilationRule):
    def __init__(self, base_rule):
        self.build_dir = base_rule.build_dir
        self.aiecc_path = base_rule.aiecc_path
        self.peano_dir = base_rule.peano_dir

    def matches(self, graph):
        return any(graph.get_worklist(comp.FullElfArtifact))

    def compile(self, graph):
        commands = []
        for artifact in graph.get_worklist(comp.FullElfArtifact):
            compile_cmd = [
                str(self.aiecc_path),
                "-v",
                "-j1",
                "--no-compile-host",
                "--no-xchesscc",
                "--no-xbridge",
                "--peano",
                str(self.peano_dir),
                "--dynamic-objFifos",
                "--expand-load-pdis",
                "--generate-full-elf",
                "--full-elf-name",
                os.path.abspath(artifact.filename),
                os.path.abspath(artifact.mlir_input.filename),
            ]
            commands.append(
                comp.ShellCompilationCommand(compile_cmd, cwd=str(self.build_dir))
            )
            artifact.available = True
        return commands


class FullELFCallable:
    def __init__(self, elf_data, device_name="main", sequence_name="sequence"):
        self.device_name = device_name
        self.sequence_name = sequence_name
        self.reload_elf(elf_data)

    def __call__(self, *args):
        run = pyxrt.run(self.xrt_kernel)
        for i, arg in enumerate(args):
            assert isinstance(arg, pyxrt.bo), f"Argument {i} is not a pyxrt.bo"
            run.set_arg(i, arg)
        run.start()
        ret_code = run.wait()
        if ret_code != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(f"Kernel execution failed with return code {ret_code}")

    def reload_elf(self, elf_data):
        self._elf_data = elf_data
        self._elf_data_u8 = self._elf_data.view(dtype=np.uint8)
        ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
        ctypes.pythonapi.PyCapsule_New.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
        ]
        capsule = ctypes.pythonapi.PyCapsule_New(
            self._elf_data_u8.ctypes.data, None, None
        )
        self._xrt_elf = pyxrt.elf(capsule, self._elf_data.nbytes)
        self._xrt_context = pyxrt.hw_context(
            aie_utils.DefaultNPURuntime._device, self._xrt_elf
        )
        self.xrt_kernel = pyxrt.ext.kernel(
            self._xrt_context, f"{self.device_name}:{self.sequence_name}"
        )


class FusedFullELFCallable(FullELFCallable):
    def __init__(self, op, elf_data=None):
        if elf_data is None:
            elf_data = load_elf(op)
        super().__init__(elf_data)

        self.op = op
        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
        self._buffers = {}
        for name in op.buffer_order:
            size = max(op.buffer_sizes[name], itemsize) // itemsize
            buffer = XRTTensor((size,), dtype=ml_dtypes.bfloat16)
            self._buffers[name] = buffer
            setattr(self, f"{name}_buffer", buffer)

        self.input_buffer = self._buffers["input"]
        self.output_buffer = self._buffers["output"]
        self.scratch_buffer = self._buffers["scratch"]
        self._buffer_cache = {}

    def get_buffer(self, buffer_name):
        if buffer_name in self._buffer_cache:
            return self._buffer_cache[buffer_name]

        buf_type, offset, length = self.op.get_layout_for_buffer(buffer_name)
        main_buffer = self._buffers[buf_type]
        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
        sub_buffer = XRTSubBuffer(
            parent_bo=main_buffer.buffer_object(),
            offset_bytes=offset,
            size_bytes=length,
            shape=(length // itemsize,),
            dtype=ml_dtypes.bfloat16,
            parent_tensor=main_buffer,
        )
        self._buffer_cache[buffer_name] = sub_buffer
        return sub_buffer

    def replace_buffer(self, name, buffer):
        if name not in self._buffers:
            raise ValueError(f"unknown fused buffer type: {name}")
        self._buffers[name] = buffer
        setattr(self, f"{name}_buffer", buffer)
        self._buffer_cache.clear()

    def __call__(self):
        self.input_buffer.to("npu")
        super().__call__(
            *[
                self._buffers[name].buffer_object()
                for name in self.op.buffer_order
            ]
        )
        self.output_buffer.to("cpu")


class FusedXclbinCallable:
    def __init__(self, op):
        self.op = op
        npu_kernel = NPUKernel(
            xclbin_path=op.xclbin_artifact.filename,
            kernel_name=op.xclbin_artifact.kernel_name,
            insts_path=op.insts_artifact.filename,
        )
        self.handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
        self._buffers = {}
        for name in op.buffer_order:
            size = max(op.buffer_sizes[name], itemsize) // itemsize
            buffer = XRTTensor((size,), dtype=ml_dtypes.bfloat16)
            self._buffers[name] = buffer
            setattr(self, f"{name}_buffer", buffer)

        self.input_buffer = self._buffers["input"]
        self.output_buffer = self._buffers["output"]
        self.scratch_buffer = self._buffers["scratch"]
        self._buffer_cache = {}

    def get_buffer(self, buffer_name):
        if buffer_name in self._buffer_cache:
            return self._buffer_cache[buffer_name]

        buf_type, offset, length = self.op.get_layout_for_buffer(buffer_name)
        main_buffer = self._buffers[buf_type]
        itemsize = np.dtype(ml_dtypes.bfloat16).itemsize
        sub_buffer = XRTSubBuffer(
            parent_bo=main_buffer.buffer_object(),
            offset_bytes=offset,
            size_bytes=length,
            shape=(length // itemsize,),
            dtype=ml_dtypes.bfloat16,
            parent_tensor=main_buffer,
        )
        self._buffer_cache[buffer_name] = sub_buffer
        return sub_buffer

    def __call__(self):
        self.input_buffer.to("npu")
        aie_utils.DefaultNPURuntime.run(
            self.handle,
            [self._buffers[name] for name in self.op.buffer_order],
        )
        self.output_buffer.to("cpu")
