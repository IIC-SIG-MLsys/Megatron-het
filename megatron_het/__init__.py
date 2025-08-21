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

# --- PRE-INSERT: put our local p2p_communication into sys.modules for vendor imports ---
def _preinsert_local_p2p():
    """
    Insert local p2p_communication module into sys.modules under the full
    name vendor code will import: 'megatron.core.pipeline_parallel.p2p_communication'.
    Must run BEFORE vendor submodules are imported.
    """
    target_name = "megatron.core.pipeline_parallel.p2p_communication"
    try:
        # Try relative import first (works when this package is installed / imported)
        try:
            from . import p2p_communication as local_p2p  # type: ignore
        except Exception:
            # Fallback to absolute import by package name
            local_p2p = importlib.import_module(f"{__name__}.p2p_communication")

        # Put our module object into sys.modules under the name vendor expects.
        sys.modules[target_name] = local_p2p

        # Also register the shorter lookup (some code may import via full path with different semantics)
        sys.modules.setdefault("megatron.core.pipeline_parallel.p2p_communication", local_p2p)

        print(f"[megatron_het] Pre-inserted local p2p_communication as {target_name}")
        return True
    except Exception as e:
        print(f"[megatron_het] Failed to pre-insert p2p shim: {e}")
        traceback.print_exc()
        return False

# Call pre-insert before vendor imports
_preinsert_local_p2p()

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