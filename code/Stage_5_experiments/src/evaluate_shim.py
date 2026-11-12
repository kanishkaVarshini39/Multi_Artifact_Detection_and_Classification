"""Thin re-export of Stage B's evaluation helpers, loaded by explicit path."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as S

_spec = importlib.util.spec_from_file_location("stage_b_evaluate",
                                               S.STAGE_B / "src" / "evaluate.py")
_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(S.STAGE_B)); sys.path.insert(0, str(S.STAGE_B / "src"))
_spec.loader.exec_module(_mod)

tune_thresholds = _mod.tune_thresholds
full_report = _mod.full_report
bh_scatter_pair = _mod.bh_scatter_pair
severity_errors = _mod.severity_errors
M = _mod.M
