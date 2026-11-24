"""Problem definitions. The engine only sees this interface:

    task.env          dict feature -> (T, N) array
    task.ops          operators allowed for this problem
    task.score(pred, split) -> float, higher is better ('train' | 'val' | 'test')

Anything that can be phrased that way can be optimised by the engine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ELEM_OPS = ["add", "sub", "mul", "div", "neg", "abs", "sq", "sqrt", "log", "exp", "inv", "sin", "cos", "tanh"]
MARKET_OPS = ["add", "sub", "mul", "div", "max", "min", "neg", "abs", "sq", "slog", "inv", "sign", "tanh",
              "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_rank", "delay", "delta", "ts_z", "ts_corr",
              "cs_rank", "cs_z"]


def _row_corr(a, b):
    """Pearson correlation per row, ignoring NaNs."""
    m = np.isfinite(a) & np.isfinite(b)
    a = np.where(m, a, np.nan)
    b = np.where(m, b, np.nan)
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    num = np.nansum(a * b, axis=1)
    den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    with np.errstate(all="ignore"):
        c = num / den
    c[m.sum(axis=1) < 10] = np.nan
    return c


class MarketTask:
    """Predict the cross-section of forward stock returns.

    Score = mean daily Spearman rank correlation (IC) between the model's value at
    the close of day t and the return from close t to close t+horizon.
    Splits are chronological with a purge gap of `horizon` days between them, so no
    target from one split overlaps the next.
    """

    name = "market"
    ops = MARKET_OPS

    def __init__(self, panel: dict, horizon=5, split=(0.6, 0.2), warmup=60, cuts=None):
        """cuts: optional (val_start, test_start, test_end) dates for walk-forward folds;
        otherwise the chronological fractions in `split` are used."""
        close, ret = panel["close"], panel["ret"]
        self.dates = close.index
        self.symbols = list(close.columns)
        A = lambda df: df.to_numpy(dtype=float)
        logc = np.log(A(close))
        vol = A(panel["volume"])
        self.env = {
            "ret": A(ret),
            "logc": logc,
            "logv": np.log(np.where(vol > 0, vol, np.nan)),
            "range": A((panel["high"] - panel["low"]) / panel["close"]),
            "gap": A(panel["open"] / panel["close"].shift(1) - 1),
            "cvwap": A(panel["close"] / panel["vwap"] - 1),
            "clpos": A((panel["close"] - panel["low"]) / (panel["high"] - panel["low"])),
        }
        # Optional non-price inputs (see rsi/extra.py).
        if "dlv" in panel:
            self.env["dlv"] = A(panel["dlv"])                      # delivery fraction
        if "oi" in panel and panel["oi"].notna().any().any():
            oi = A(panel["oi"])
            basis = A(panel["basis"])
            if np.isfinite(basis).mean() > 0.6:   # most stocks have futures (Nifty 100)
                self.env["loi"] = np.log(np.where(oi > 0, oi, np.nan))  # log futures open interest
                self.env["basis"] = basis                               # front future / spot - 1
            else:                                 # sparse (Nifty 500): 0 = no futures signal
                self.env["basis"] = np.nan_to_num(basis)
        if "vix" in panel:
            self.env["lvix"] = np.log(A(panel["vix"]))              # market fear gauge
        # Fundamentals (rsi/fundamentals.py) and NSE daily files (rsi/market_features.py).
        for k in ["ey", "rev_g", "prof_g", "fresh", "var", "fii_fut", "cli_fut", "fii_stk", "bulk",
                  "ins_net", "pledge", "prom_chg", "ann_surp", "ann_call", "ann_deal",
                  "sec_rel", "sec_rel5", "sec_mom", "mkt20", "small_big", "mtf", "mtf_chg", "pcr", "opt_act",
                  "sty_mom", "sty_lowvol", "sty_qual", "sty_val", "sty_eqw",
                  "grp_rel5", "grp_rel20", "grp_mom", "mem_rel5", "mem_mom"]:
            if k in panel and np.isfinite(A(panel[k])).mean() > 0.3:
                self.env[k] = A(panel[k])
        fwd = np.full_like(logc, np.nan)
        fwd[:-horizon] = logc[horizon:] - logc[:-horizon]
        self.target = fwd
        self._target_rank = pd.DataFrame(fwd).rank(axis=1).to_numpy()
        self.horizon = horizon
        T = len(self.dates)
        usable = np.arange(warmup, T - horizon)
        if cuts is None:
            n = len(usable)
            a, b = int(n * split[0]), int(n * (split[0] + split[1]))
            c = len(usable)
        else:
            pos = lambda d: int(np.searchsorted(self.dates[usable], pd.Timestamp(d)))
            a, b, c = pos(cuts[0]), pos(cuts[1]), pos(cuts[2])
        # Purge `horizon` days before each boundary so no target overlaps the next split.
        self.splits = {
            "train": usable[:max(0, a - horizon)],
            "val": usable[a:max(a, b - horizon)],
            "test": usable[b:c],
        }

    def describe(self):
        d = self.dates
        return {k: f"{d[v[0]].date()} -> {d[v[-1]].date()} ({len(v)} days)" for k, v in self.splits.items()}

    def ic_series(self, pred, split):
        rows = self.splits[split]
        p = pd.DataFrame(pred[rows]).rank(axis=1).to_numpy()
        return _row_corr(p, self._target_rank[rows])

    def score(self, pred, split):
        rows = self.splits[split]
        if np.isfinite(pred[rows]).mean() < 0.5:
            return -1.0
        ic = self.ic_series(pred, split)
        if np.isfinite(ic).mean() < 0.5:
            return -1.0
        if split != "train":
            return float(np.nanmean(ic))
        # Robust training fitness: half mean IC, half the WORST of 4 sub-periods, so a
        # formula that only worked in one market regime cannot win selection.
        chunks = [np.nanmean(c) for c in np.array_split(ic, 4)]
        return float(0.5 * np.nanmean(ic) + 0.5 * min(chunks))

    def portfolio(self, pred, split, cost_bps=10.0):
        """Daily long-short: long top fifth, short bottom fifth, held `horizon` days
        (1/horizon of the book rebalanced each day). Returns daily net return series."""
        rows = self.splits[split]
        r = pd.DataFrame(pred[rows]).rank(axis=1, pct=True).to_numpy()
        w = np.where(r >= 0.8, 1.0, 0.0) - np.where(r <= 0.2, 1.0, 0.0)
        w = np.nan_to_num(w)
        n = np.maximum(np.abs(w).sum(axis=1, keepdims=True), 1)
        w = w / n  # gross exposure 1
        # Average of the last `horizon` daily books (overlapping holdings).
        w = pd.DataFrame(w).rolling(self.horizon, min_periods=1).mean().to_numpy()
        nxt = np.nan_to_num(self.env["ret"][rows + 1])
        gross = (w * nxt).sum(axis=1)
        turnover = np.abs(np.diff(w, axis=0, prepend=0)).sum(axis=1)
        return pd.Series(gross - turnover * cost_bps / 1e4, index=self.dates[rows])


class RegressionTask:
    """Plain symbolic regression: find f(X) ~ y. Score = R^2 after a linear
    rescaling a + b*f fitted on the training rows only."""

    name = "regression"
    ops = ELEM_OPS

    def __init__(self, X: dict, y, split=(0.6, 0.2), seed=0, name="regression"):
        self.name = name
        self.env = {k: np.asarray(v, float).reshape(-1, 1) for k, v in X.items()}
        self.target = np.asarray(y, float).reshape(-1, 1)
        idx = np.random.default_rng(seed).permutation(len(y))
        a, b = int(len(y) * split[0]), int(len(y) * (split[0] + split[1]))
        self.splits = {"train": idx[:a], "val": idx[a:b], "test": idx[b:]}

    def describe(self):
        return {k: f"{len(v)} rows" for k, v in self.splits.items()}

    def _scale(self, pred):
        r = self.splits["train"]
        f, y = pred[r, 0], self.target[r, 0]
        m = np.isfinite(f)
        if m.mean() < 0.95 or np.nanstd(f[m]) < 1e-12:
            return None
        b = np.cov(f[m], y[m])[0, 1] / np.var(f[m])
        return y[m].mean() - b * f[m].mean(), b

    def score(self, pred, split):
        s = self._scale(pred)
        if s is None:
            return -1.0
        r = self.splits[split]
        f, y = s[0] + s[1] * pred[r, 0], self.target[r, 0]
        if not np.all(np.isfinite(f)):
            return -1.0
        return float(max(-1.0, 1 - np.mean((f - y) ** 2) / np.var(y)))


def physics_family(seed=0, n=600, noise=0.01):
    """A family of related hidden laws, used to test whether skills learned on some
    problems transfer to new ones. Each entry: (name, f(x1,x2,x3))."""
    laws = [
        ("gravity", lambda a, b, c: a * b / (c * c)),
        ("kinetic", lambda a, b, c: 0.5 * a * b * b),
        ("pendulum", lambda a, b, c: 2 * np.pi * np.sqrt(a / b)),
        ("damped", lambda a, b, c: np.exp(-a * c) * np.cos(b * c)),
        ("coulomb_pair", lambda a, b, c: a * b / (c * c) + a / c),
        ("relativistic", lambda a, b, c: a / np.sqrt(1 + (b / (c + 2)) ** 2)),
        ("wave_energy", lambda a, b, c: a * b * b * np.sin(c) ** 2),
        ("lens", lambda a, b, c: a * b / (a + b)),
    ]
    rng = np.random.default_rng(seed)
    out = []
    for name, f in laws:
        X = {"x1": rng.uniform(0.5, 3, n), "x2": rng.uniform(0.5, 3, n), "x3": rng.uniform(0.5, 3, n)}
        y = f(X["x1"], X["x2"], X["x3"])
        y = y + noise * np.std(y) * rng.standard_normal(n)
        out.append(RegressionTask(X, y, seed=seed, name=name))
    return out
