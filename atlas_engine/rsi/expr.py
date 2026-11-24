"""Expression trees: the 'mathematical model' language.

A model is a tree of primitives evaluated over 2-D arrays shaped (T, N):
T = time steps / samples, N = cross-section (stocks) or 1 for plain regression.

Leaf kinds:
  feat  - an input variable (param = feature name)
  const - a numeric constant (param = float)
  macro - a learned sub-model from the library (param = macro id)
Internal nodes are operators from OPS. Time-series operators carry a window (param = int).

The language is open-ended: the engine adds macros to it while it runs.
"""
from __future__ import annotations
import random
import numpy as np
import pandas as pd

WINDOWS = [2, 3, 5, 10, 20, 40, 60]


def _safe(x):
    x = np.array(x, dtype=float, copy=True)
    x[~np.isfinite(x)] = np.nan
    return np.clip(x, -1e12, 1e12)


def _roll(x, w, how):
    r = pd.DataFrame(x).rolling(w, min_periods=max(2, w // 2))
    return getattr(r, how)().to_numpy()


def _ts_rank(x, w):
    return pd.DataFrame(x).rolling(w, min_periods=max(2, w // 2)).rank(pct=True).to_numpy()


def _ts_corr(a, b, w):
    return pd.DataFrame(a).rolling(max(w, 3), min_periods=max(3, w // 2)).corr(pd.DataFrame(b)).to_numpy()


def _shift(x, w):
    out = np.full_like(x, np.nan)
    out[w:] = x[:-w]
    return out


def _cs_rank(x):
    return pd.DataFrame(x).rank(axis=1, pct=True).to_numpy() - 0.5


def _cs_z(x):
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / np.where(s > 0, s, np.nan)


# name: (arity, kind, function). kind: elem | ts | cs
OPS = {
    "add": (2, "elem", lambda a, b: a + b),
    "sub": (2, "elem", lambda a, b: a - b),
    "mul": (2, "elem", lambda a, b: a * b),
    "div": (2, "elem", lambda a, b: a / np.where(np.abs(b) > 1e-9, b, np.nan)),
    "max": (2, "elem", np.fmax),
    "min": (2, "elem", np.fmin),
    "neg": (1, "elem", lambda a: -a),
    "abs": (1, "elem", np.abs),
    "sq": (1, "elem", lambda a: a * a),
    "sqrt": (1, "elem", lambda a: np.sqrt(np.abs(a))),
    "slog": (1, "elem", lambda a: np.sign(a) * np.log1p(np.abs(a))),
    "log": (1, "elem", lambda a: np.log(np.abs(a) + 1e-12)),
    "exp": (1, "elem", lambda a: np.exp(np.clip(a, -30, 30))),
    "inv": (1, "elem", lambda a: 1.0 / np.where(np.abs(a) > 1e-9, a, np.nan)),
    "sin": (1, "elem", np.sin),
    "cos": (1, "elem", np.cos),
    "tanh": (1, "elem", np.tanh),
    "sign": (1, "elem", np.sign),
    "ts_mean": (1, "ts", lambda a, w: _roll(a, w, "mean")),
    "ts_std": (1, "ts", lambda a, w: _roll(a, w, "std")),
    "ts_max": (1, "ts", lambda a, w: _roll(a, w, "max")),
    "ts_min": (1, "ts", lambda a, w: _roll(a, w, "min")),
    "ts_sum": (1, "ts", lambda a, w: _roll(a, w, "sum")),
    "ts_rank": (1, "ts", _ts_rank),
    "delay": (1, "ts", _shift),
    "delta": (1, "ts", lambda a, w: a - _shift(a, w)),
    "ts_z": (1, "ts", lambda a, w: (a - _roll(a, w, "mean")) / _roll(a, w, "std")),
    "ts_corr": (2, "ts", _ts_corr),
    "cs_rank": (1, "cs", _cs_rank),
    "cs_z": (1, "cs", _cs_z),
}


class Node:
    __slots__ = ("op", "kids", "param")

    def __init__(self, op, kids=(), param=None):
        self.op, self.kids, self.param = op, list(kids), param

    # ---- structure -------------------------------------------------------
    def copy(self):
        return Node(self.op, [k.copy() for k in self.kids], self.param)

    def nodes(self):
        out = [self]
        for k in self.kids:
            out.extend(k.nodes())
        return out

    def size(self, macro_cost=None):
        if self.op == "macro":
            return 1 if macro_cost is None else macro_cost
        return 1 + sum(k.size(macro_cost) for k in self.kids)

    def depth(self):
        return 1 + max((k.depth() for k in self.kids), default=0)

    def primitives(self):
        """Names of every primitive used (operators, features, macros)."""
        if self.op == "feat":
            return [f"feat:{self.param}"]
        if self.op == "macro":
            return [f"macro:{self.param}"]
        if self.op == "const":
            return ["const"]
        return [self.op] + [p for k in self.kids for p in k.primitives()]

    # ---- serialisation ---------------------------------------------------
    def key(self) -> str:
        if self.op == "feat":
            return str(self.param)
        if self.op == "const":
            return f"{self.param:.4g}"
        if self.op == "macro":
            return f"M{self.param}"
        p = f"{self.param}," if self.param is not None else ""
        return f"{self.op}({p}{','.join(k.key() for k in self.kids)})"

    __repr__ = key

    def to_json(self):
        return [self.op, self.param, [k.to_json() for k in self.kids]]

    @staticmethod
    def from_json(j):
        return Node(j[0], [Node.from_json(k) for k in j[2]], j[1])


def evaluate(node: Node, env: dict, cache: dict, macros: dict):
    """Evaluate a tree. env: feature name -> (T,N) array. cache: key -> array (shared)."""
    k = node.key()
    if k in cache:
        return cache[k]
    if node.op == "feat":
        v = env[node.param]
    elif node.op == "const":
        v = np.full_like(next(iter(env.values())), node.param, dtype=float)
    elif node.op == "macro":
        v = evaluate(macros[node.param], env, cache, macros)
    else:
        arity, kind, fn = OPS[node.op]
        args = [evaluate(c, env, cache, macros) for c in node.kids]
        with np.errstate(all="ignore"):
            v = _safe(fn(*args, node.param) if kind == "ts" else fn(*args))
    # Only cache subtrees that are cheap to keep around: leaves, macros, and small nodes.
    if node.op in ("feat", "macro") or len(cache) < getattr(cache, "limit", 4000):
        cache[k] = v
    return v


class Grammar:
    """Which primitives are available, with learnable selection weights."""

    def __init__(self, features, allowed_ops, const_range=(-2.0, 2.0)):
        self.features = list(features)
        self.ops = [o for o in allowed_ops if o in OPS]
        self.const_range = const_range
        self.weights = {name: 1.0 for name in self.ops + [f"feat:{f}" for f in self.features] + ["const"]}

    def add_macro(self, mid):
        self.weights.setdefault(f"macro:{mid}", 1.5)

    def remove_macro(self, mid):
        self.weights.pop(f"macro:{mid}", None)

    def _pick(self, names, rng):
        w = np.array([self.weights.get(n, 1.0) for n in names])
        return names[rng.choices(range(len(names)), weights=w)[0]]

    def random_leaf(self, rng):
        leaves = [n for n in self.weights if n.startswith(("feat:", "macro:")) or n == "const"]
        name = self._pick(leaves, rng)
        if name == "const":
            return Node("const", param=round(rng.uniform(*self.const_range), 3))
        kind, val = name.split(":", 1)
        return Node(kind, param=int(val) if kind == "macro" else val)

    def random_op_node(self, rng, kids_fn):
        name = self._pick(self.ops, rng)
        arity, kind, _ = OPS[name]
        param = rng.choice(WINDOWS) if kind == "ts" else None
        return Node(name, [kids_fn() for _ in range(arity)], param)

    def random_tree(self, rng, max_depth, full=False):
        if max_depth <= 1 or (not full and rng.random() < 0.3):
            return self.random_leaf(rng)
        return self.random_op_node(rng, lambda: self.random_tree(rng, max_depth - 1, full))

    def reinforce(self, names, amount, decay=0.98, floor=0.2, cap=5.0):
        """Credit assignment: primitives present in successful offspring gain weight.
        Weights are renormalised to mean 1 and bounded, so they express *relative*
        preference and can't inflate without limit."""
        for n in set(names):
            if n in self.weights:
                self.weights[n] += amount
        mean = sum(self.weights.values()) / len(self.weights)
        for n in self.weights:
            w = self.weights[n] / mean
            self.weights[n] = min(cap, max(floor, w * decay + (1 - decay)))
