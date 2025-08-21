import os
import sys
import importlib
import traceback
from types import ModuleType
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRDPARTY_ROOT = os.path.join(PROJECT_ROOT, "thirdparty")

VENDOR_PATHS = {
    "ascend": os.path.join(THIRDPARTY_ROOT, "ascend_megatron_lm"),
    "nvidia": os.path.join(THIRDPARTY_ROOT, "nvidia_megatron_lm_0_9_0"),
    "mlu": os.path.join(THIRDPARTY_ROOT, "mlu_megatron_lm"),
}

_PROVIDER = os.environ.get("MEGATRON_PROVIDER", "nvidia").lower()
if _PROVIDER not in VENDOR_PATHS:
    raise ValueError(f"Invalid MEGATRON_PROVIDER='{_PROVIDER}'")

_SELECTED_ROOT = VENDOR_PATHS[_PROVIDER]
_VENDOR_MEGA_PATH = os.path.join(_SELECTED_ROOT, "megatron")

if not os.path.isdir(_SELECTED_ROOT):
    raise ImportError(f"Vendor root not found: {_SELECTED_ROOT}")

if _SELECTED_ROOT not in sys.path:
    sys.path.insert(0, _SELECTED_ROOT)

def _import_submodule(name: str) -> Optional[ModuleType]:
    full_name = f"megatron.{name}"
    try:
        return importlib.import_module(full_name)
    except Exception as e:
        print(f"[megatron_het] import {full_name} failed: {e}")
        traceback.print_exc(limit=1)
        return None

core = _import_submodule("core")
training = _import_submodule("training")
legacy = _import_submodule("legacy")

if core is not None:
    sys.modules[f"{__name__}.core"] = core
if training is not None:
    sys.modules[f"{__name__}.training"] = training
if legacy is not None:
    sys.modules[f"{__name__}.legacy"] = legacy

def get_vendor_info():
    return {
        "provider": _PROVIDER,
        "selected_root": _SELECTED_ROOT,
        "vendor_megatron_path": _VENDOR_MEGA_PATH,
        "vendor_present": os.path.isdir(_VENDOR_MEGA_PATH),
        "core_loaded": core is not None,
        "training_loaded": training is not None,
        "legacy_loaded": legacy is not None,
    }

__all__ = ["get_vendor_info", "core", "training", "legacy"]

# # --- replace p2p_communication ---
# try:
#     from . import p2p_communication as my_p2
#     sys.modules["megatron.core.pipeline_parallel.p2p_communication"] = my_p2
#     print("[megatron init] p2p_communication overridden with local implementation")
# except Exception as e:
#     print(f"[megatron init] Failed to override p2p_communication: {e}")
#     traceback.print_exc()