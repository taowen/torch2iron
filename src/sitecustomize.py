"""Project-local Python startup fixes for the AMD NPU/XRT stack."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_xrt_python_path() -> None:
    candidates = []
    if "XILINX_XRT" in os.environ:
        candidates.append(Path(os.environ["XILINX_XRT"]))
    candidates.extend((Path("/var/opt/xilinx/xrt"), Path("/opt/xilinx/xrt")))

    for xrt_root in candidates:
        python_dir = xrt_root / "python"
        if python_dir.exists():
            os.environ.setdefault("XILINX_XRT", str(xrt_root))
            python_path = str(python_dir)
            if python_path not in sys.path:
                sys.path.insert(0, python_path)
            return


_add_xrt_python_path()
