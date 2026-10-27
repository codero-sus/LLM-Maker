"""Import the generated ``train.py`` without a working PyTorch install.

The generated scripts are ordinary PyTorch code, but the *data pipeline* inside
them is plain Python: field detection, chat-template rendering, prompt masking,
preference-pair splitting and sequence packing touch no tensors.  Those functions
are the easiest part of a training run to get subtly wrong, so they deserve real
tests.

Torch is a multi-gigabyte install, so when it is genuinely absent this module
installs a minimal stand-in covering only what ``train.py`` touches *at import
time* (the module object, ``torch.nn.functional``, and a couple of dtypes).
Anything that actually needs tensors is skipped via ``torch_available()``.

This is a test shim, not a torch implementation.  Nothing here should be used
outside the test suite.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path


def torch_available() -> bool:
    """True when a real PyTorch is importable."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001 - broken/partial installs count as absent
        return False
    return bool(getattr(torch, "__version__", None))


class _AnyTorch:
    """A permissive stand-in: callable, subscriptable, attribute-returning.

    ``datasets``' pickling layer probes torch object types on the way past, and
    the exact set of attributes it touches varies between versions. Rather than
    chase that list, anything the real torch would have but the stub does not is
    handed back as one of these. Nothing in the tests under this shim calls into
    torch, so an accidental interaction fails loudly elsewhere rather than
    silently returning a plausible value.
    """

    def __call__(self, *args, **kwargs) -> "_AnyTorch":
        return self

    def __getattr__(self, name: str) -> "_AnyTorch":
        return self

    def __getitem__(self, item) -> "_AnyTorch":
        return self

    def __iter__(self):
        return iter(())

    def __mro_entries__(self, bases):
        return (object,)

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<torch stub>"


def _install_stub() -> None:
    import importlib.machinery

    torch = types.ModuleType("torch")
    # A module in sys.modules without a __spec__ makes importlib.util.find_spec
    # raise deep inside transformers, so give it a real one.
    torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    torch.__version__ = "0.0.0-torch-stub-for-llm-maker-tests"
    torch.__path__ = []

    torch.bfloat16 = _AnyTorch()
    torch.float16 = _AnyTorch()
    torch.float32 = _AnyTorch()
    torch.long = _AnyTorch()
    torch.Tensor = type("Tensor", (), {})

    class _NullContext:
        """Works as both a context manager and a decorator, like the real ones."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, function):
            return function

    torch.no_grad = lambda *args, **kwargs: _NullContext()
    torch.inference_mode = lambda *args, **kwargs: _NullContext()

    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = type("Module", (), {})
    functional = types.ModuleType("torch.nn.functional")
    functional.cross_entropy = _AnyTorch()
    functional.log_softmax = _AnyTorch()
    functional.logsigmoid = _AnyTorch()
    torch.nn.functional = functional

    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    cuda.device_count = lambda: 0
    cuda.is_bf16_supported = lambda: False
    torch.cuda = cuda

    def _missing(name: str) -> _AnyTorch:
        value = _AnyTorch()
        setattr(torch, name, value)
        return value

    # PEP 562: only consulted when a normal lookup fails, so the explicit
    # definitions above still win.
    torch.__getattr__ = _missing

    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", torch.nn)
    sys.modules.setdefault("torch.nn.functional", functional)
    sys.modules.setdefault("torch.cuda", cuda)


_LOADED: dict[str, types.ModuleType] = {}


def _prime_heavy_deps() -> None:
    """Import transformers/datasets *before* any torch stub exists.

    They probe for torch at import time; doing this first means the probe sees a
    truthful "no torch" answer instead of a stand-in it might try to use.
    """
    for name in ("transformers", "datasets"):
        try:
            importlib.import_module(name)
        except ImportError:  # pragma: no cover - the tests skip without them
            pass


def load_generated_train(project_dir: str | Path) -> types.ModuleType:
    """Import the ``train.py`` from a generated project directory."""
    _prime_heavy_deps()
    directory = Path(project_dir)
    key = str(directory.resolve())
    if key in _LOADED:
        return _LOADED[key]

    if not torch_available():
        _install_stub()

    module_name = f"generated_train_{abs(hash(key)) % 10**8}"
    spec = importlib.util.spec_from_file_location(module_name, directory / "train.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {directory / 'train.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _LOADED[key] = module
    return module
