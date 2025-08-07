#!/usr/bin/env python3
"""
Run test_all_models.py with global NaN injection and capture all output
into trace_withNA_output.txt (same format as test_results_* file).

Usage:  python run_all_models_with_nan.py
"""

import os
import sys
import pathlib
import runpy
import torch.nn as nn

LD_PATH = "/home/ganesh/nixnan.so"                 # nixnan library
BASE    = pathlib.Path("/home/jiesi/pytorch_GPU")  # project root
SCRIPT_DIR = pathlib.Path(__file__).parent         # withNA folder
TRACE   = SCRIPT_DIR / "trace_withNA_output.txt"   # destination file

# ──────────────────────────────────────────────────────────────
# 0) on first entry → set LD_PRELOAD, re-exec ourself
# ──────────────────────────────────────────────────────────────
if "LD_PRELOAD" not in os.environ:
    os.environ["LD_PRELOAD"] = LD_PATH
    os.execv(sys.executable, [sys.executable, *sys.argv])

# ──────────────────────────────────────────────────────────────
# 1) redirect *both* Python-level and C-level stdout / stderr
# ──────────────────────────────────────────────────────────────
# open file for (truncate+)append, mode 644
fd = os.open(TRACE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
# duplicate to stdout (1) and stderr (2)
os.dup2(fd, 1)
os.dup2(fd, 2)
# refresh sys stdout/err objects
sys.stdout = os.fdopen(1, "w", buffering=1)
sys.stderr = os.fdopen(2, "w", buffering=1)

# ──────────────────────────────────────────────────────────────
# 2) import NaN-decorator and wrap every nn.Module.forward
# ──────────────────────────────────────────────────────────────
sys.path.append(str(SCRIPT_DIR))     # allow importing NA_inject.py from current dir
from NA_inject import inject        # noqa: E402

nn.Module.forward = inject(
    corruption_probability=0.05,    # 5 % of elements corrupted
    nan_frac=1.0                    # all corruptions become NaN
)(nn.Module.forward)

print("🧪  Running test_all_models.py with NaN injection")
print(f"Output is being captured in {TRACE}")
print("==========================================================")

# ──────────────────────────────────────────────────────────────
# 3) execute the original test suite
# ──────────────────────────────────────────────────────────────
runpy.run_path(str(BASE / "noNA" / "test_all_models.py"), run_name="__main__")

print("\n✅  Finished. Full trace saved.")