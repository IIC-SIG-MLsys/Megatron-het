"""
Wrapper: 封装对 Megatron-LM 的 vendor 依赖，支持 Ascend / NVIDIA 版本切换。
通过环境变量 MEGATRON_PROVIDER 控制（"ascend" 或 "nvidia"）。
确保 megatron.core 和 megatron.training 可通过本包导入。
"""

import os
import sys
import importlib
import importlib.util
import traceback

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
thirdparty_root = os.path.join(project_root, "thirdparty")

VENDOR_PATHS = {
    "ascend": os.path.join(thirdparty_root, "ascend_megatron_lm"),
    "nvidia": os.path.join(thirdparty_root, "nvidia_megatron_lm_0_9_0"),
    "mlu": os.path.join(thirdparty_root, "mlu_megatron_lm"),
}

_selected_key = os.environ.get("MEGATRON_PROVIDER", "nvidia").lower()
if _selected_key not in VENDOR_PATHS:
    raise ValueError(f"Invalid MEGATRON_PROVIDER='{_selected_key}'. Must be one of {list(VENDOR_PATHS.keys())}")

selected_root = VENDOR_PATHS[_selected_key]
vendor_pkg_dir = os.path.join(selected_root, "megatron")

if not os.path.isdir(vendor_pkg_dir):
    raise ImportError(
        f"[megatron init] Vendor package not found: {vendor_pkg_dir}\n"
        f"Please check:\n"
        f"  - Path exists\n"
        f"  - Run 'git submodule update --init' (if using submodules)\n"
        f"  - Or manually clone into {selected_root}"
    )

if vendor_pkg_dir not in __path__:
    __path__.insert(0, vendor_pkg_dir)

if 'megatron' not in sys.modules:
    sys.modules['megatron'] = sys.modules[__name__]

core = None
training = None

def _try_import(name, pkg_name, file_hint=None):
    """
    尝试导入模块，优先用 importlib.import_module，失败则尝试从文件加载。
    file_hint 用于捕获执行时错误（如缺失 transformer-engine）。
    """
    try:
        mod = importlib.import_module(name, pkg_name)
        return mod
    except Exception as e:
        print(f"[megatron init] FAILED to import {pkg_name}.{name} via import_module: {e}")
        if file_hint and os.path.isfile(file_hint):
            print(f"    → File exists: {file_hint}, attempting direct load...")
        else:
            print(f"    → File not found: {file_hint}")
        traceback.print_exc(limit=1)
        print("")

    if file_hint and os.path.isfile(file_hint):
        try:
            spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", file_hint)
            if spec is None:
                print(f"[megatron init] Failed to create spec for {file_hint}")
                return None
            mod = importlib.util.module_from_spec(spec)
            # 注册到 sys.modules 避免重复加载
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            print(f"[megatron init] ✅ Loaded {pkg_name}.{name} from file: {file_hint}")
            return mod
        except Exception as e:
            print(f"[megatron init] ❌ Failed to execute {file_hint}: {e}")
            traceback.print_exc()
            return None

    return None


_core_init = os.path.join(vendor_pkg_dir, "core", "__init__.py")
core = _try_import("core", __name__, _core_init)
if core is not None:
    globals()["core"] = core

_training_init = os.path.join(vendor_pkg_dir, "training", "__init__.py")
training = _try_import("training", __name__, _training_init)
if training is not None:
    globals()["training"] = training

print(f"[megatron init] 🚀 Vendor: {_selected_key} → {selected_root}")
print(f"[megatron init] 📂 Package: {vendor_pkg_dir}")
print(f"[megatron init] ✅ core loaded:    {core is not None}")
print(f"[megatron init] ✅ training loaded: {training is not None}")

try:
    import subprocess
    label = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=selected_root,
        stderr=subprocess.DEVNULL
    ).decode().strip()
    print(f"[megatron init] 🔖 Git commit: {label}")
except Exception:
    print("[megatron init] 🔖 Git commit: unknown")

# --- from megatron import * ---
__all__ = []
if core is not None:
    __all__.append("core")
if training is not None:
    __all__.append("training")


def get_vendor_info():
    return {
        "provider": _selected_key,
        "path": selected_root,
        "core_loaded": core is not None,
        "training_loaded": training is not None,
    }


# --- replace p2p_communication ---
try:
    from . import p2p_communication as my_p2p

    # sys.modules 路径要和第三方 scheduler 里 import 的路径一致
    sys.modules["megatron.core.pipeline_parallel.p2p_communication"] = my_p2p

    print("[megatron init] ✅ p2p_communication overridden with local implementation")

except Exception as e:
    print(f"[megatron init] ❌ Failed to override p2p_communication: {e}")
    traceback.print_exc()
