#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

import aie.utils.config as config


def compile_env() -> dict[str, str]:
    env = os.environ.copy()
    candidates = [
        Path(env.get("XILINX_XRT", "")) / "bin",
        Path("/var/opt/xilinx/xrt/bin"),
        Path("/opt/xilinx/xrt/bin"),
        Path("/var/home/taowen/projects/xdna-driver/xrt/build/Release/opt/xilinx/xrt/bin"),
    ]
    xrt_bins = [str(path) for path in candidates if path.is_dir()]
    if xrt_bins:
        env["PATH"] = os.pathsep.join(xrt_bins + [env.get("PATH", "")])
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlir", type=Path)
    parser.add_argument("--xclbin", type=Path, required=True)
    parser.add_argument("--kernel", default="MLIR_AIE")
    args = parser.parse_args()

    aiecc = Path(config.root_path()) / "bin" / "aiecc"
    peano = Path(config.peano_install_dir())
    args.xclbin.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(aiecc),
        "-v",
        "-j1",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        "--peano",
        str(peano),
        "--dynamic-objFifos",
        "--aie-generate-xclbin",
        f"--xclbin-name={args.xclbin.resolve()}",
        f"--xclbin-kernel-name={args.kernel}",
        str(args.mlir.resolve()),
    ]
    subprocess.run(cmd, check=True, env=compile_env())


if __name__ == "__main__":
    main()
