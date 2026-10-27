"""Benchmark registry. Adding a benchmark is one decorator plus a function."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Sequence

BENCHMARKS: "OrderedDict[str, Callable[[Any], List]]" = OrderedDict()
BENCHMARK_META: Dict[str, Dict[str, Any]] = {}


def register(name: str, datasets: Sequence[str] = ("*",),
             requires: Sequence[str] = (), default: bool = True):
    """Register a benchmark.

    ``datasets`` limits it to particular datasets ("*" for any), ``requires``
    names keys that must be present in ``ctx.extra`` (checked before the call,
    so a missing classifier fails with a readable message rather than a
    KeyError three frames down).
    """
    def deco(fn):
        if name in BENCHMARKS:
            raise ValueError("benchmark {!r} already registered".format(name))
        BENCHMARKS[name] = fn
        BENCHMARK_META[name] = {"datasets": tuple(datasets),
                                "requires": tuple(requires),
                                "default": bool(default),
                                "doc": (fn.__doc__ or "").strip().split("\n")[0]}
        return fn
    return deco


def applicable(dataset: str, only_default: bool = True) -> List[str]:
    out = []
    for name, meta in BENCHMARK_META.items():
        if only_default and not meta["default"]:
            continue
        if "*" in meta["datasets"] or dataset in meta["datasets"]:
            out.append(name)
    return out
