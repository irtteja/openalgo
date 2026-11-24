"""The self-improving search engine.

Three nested loops, each improving the one inside it:

  1. Model level    - genetic programming evolves expression trees (models).
  2. Search level   - the engine learns HOW to search:
                        * credit assignment: variation operators and primitives that
                          produce improved offspring get chosen more often;
                        * population-based training: islands run with different
                          hyperparameters; weak islands copy and perturb strong ones.
  3. Language level - library learning: sub-expressions that recur across the best
                      models are promoted to new primitives ("macros"). Macros can be
                      built from earlier macros, so the language grows recursively,
                      and each promotion compresses models, letting later search
                      reach deeper structure at the same complexity cost.

All learned state (library, weights, hyperparameters, hall of fame, history) is
saved to JSON so each run starts from what previous runs learned.
"""
from __future__ import annotations
import json, math, os, random, time
from collections import Counter, defaultdict
import numpy as np

from .expr import Node, Grammar, OPS, WINDOWS, evaluate

VAR_OPS = ["subtree", "point", "param", "crossover", "hoist", "wrap", "macro"]


class Cache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def trim(self):
        for k in [k for k in self if "(" in k]:
            del self[k]


class Individual:
    __slots__ = ("tree", "train", "val", "fit", "key")

    def __init__(self, tree, train=-1.0, val=-1.0, fit=-1e9):
        self.tree, self.train, self.val, self.fit = tree, train, val, fit
        self.key = tree.key()


class Island:
    def __init__(self, hyper, rng):
        self.hyper = dict(hyper)
        self.var_rate = {v: 0.2 for v in VAR_OPS}  # EMA of each operator's success rate
        self.pop: list[Individual] = []
        self.rng = rng
        self.best_seen = -1e9
        self.stagnant = 0

    def pick_var(self, has_library):
        names = [v for v in VAR_OPS if has_library or v != "macro"]
        w = [max(0.03, self.var_rate[v]) for v in names]
        return self.rng.choices(names, weights=w)[0]


class Engine:
    def __init__(self, task, seed=0, n_islands=4, pop_size=60, state_path=None, log=print):
        self.task, self.log = task, log
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.grammar = Grammar(task.env.keys(), task.ops)
        self.macros: dict[int, Node] = {}
        self.macro_meta: dict[int, dict] = {}
        self.next_macro = 0
        self.pop_size = pop_size
        base = dict(tournament=5, parsimony=0.0015, max_depth=6, init_depth=4)
        self.islands = [Island(self._perturb(base) if i else base, random.Random(seed * 100 + i))
                        for i in range(n_islands)]
        sample = next(iter(task.env.values()))
        self.cache = Cache(limit=max(200, int(3e8 / max(sample.nbytes, 1))))
        self.score_cache: dict[str, tuple] = {}
        self.hof: list[Individual] = []
        self.history: list[dict] = []
        self.evals = 0
        self.epoch = 0
        # Complexity penalties are expressed in units of this task's typical score, so the
        # same settings work for R^2 (~1) and market IC (~0.03). Set in run().
        self.scale = 1.0
        self.state_path = state_path
        if state_path and os.path.exists(state_path):
            self._load(state_path)

    # ------------------------------------------------------------------ utils
    def _perturb(self, h):
        r = self.rng
        h = dict(h)
        h["tournament"] = int(min(12, max(2, h["tournament"] + r.choice([-1, 0, 1]))))
        h["parsimony"] = float(min(0.02, max(2e-4, h["parsimony"] * r.choice([0.6, 0.8, 1.25, 1.6]))))
        h["max_depth"] = int(min(9, max(3, h["max_depth"] + r.choice([-1, 0, 1]))))
        h["init_depth"] = int(min(h["max_depth"], max(2, h["init_depth"] + r.choice([-1, 0, 1]))))
        return h

    def expand(self, node: Node) -> Node:
        """Replace macros with their definitions (fully)."""
        if node.op == "macro":
            return self.expand(self.macros[node.param])
        return Node(node.op, [self.expand(k) for k in node.kids], node.param)

    def macro_level(self, mid):
        refs = [n.param for n in self.macros[mid].nodes() if n.op == "macro"]
        return 1 + max((self.macro_level(r) for r in refs), default=0)

    def estimate_scale(self, n=200):
        """Typical achievable score on this task: the 90th percentile of |train score|
        over random models. Makes complexity penalties scale-free across tasks."""
        rng = random.Random(12345)
        vals = []
        for _ in range(n):
            t = self.grammar.random_tree(rng, rng.randint(2, 4))
            if t.key() not in self.score_cache:
                pred = evaluate(t, self.task.env, self.cache, self.macros)
                self.score_cache[t.key()] = (self.task.score(pred, "train"), self.task.score(pred, "val"))
            tr = self.score_cache[t.key()][0]
            if tr > -1:
                vals.append(abs(tr))
        return float(max(1e-3, np.percentile(vals, 90))) if vals else 1.0

    # --------------------------------------------------------------- evaluate
    def score(self, tree: Node, island: Island) -> Individual:
        ind = Individual(tree)
        if ind.key not in self.score_cache:
            pred = evaluate(tree, self.task.env, self.cache, self.macros)
            self.evals += 1
            self.score_cache[ind.key] = (self.task.score(pred, "train"), self.task.score(pred, "val"))
        ind.train, ind.val = self.score_cache[ind.key]
        ind.fit = ind.train - self.scale * island.hyper["parsimony"] * tree.size(macro_cost=2)
        return ind

    # -------------------------------------------------------------- variation
    def _random_node(self, tree, rng):
        nodes = tree.nodes()
        return nodes[rng.randrange(len(nodes))]

    def _replace(self, tree, target, new):
        if tree is target:
            return new
        tree.kids = [self._replace(k, target, new) for k in tree.kids]
        return tree

    def vary(self, op, a: Node, b: Node, island: Island) -> Node:
        rng, g = island.rng, self.grammar
        t = a.copy()
        n = self._random_node(t, rng)
        if op == "subtree":
            t = self._replace(t, n, g.random_tree(rng, rng.randint(1, 3)))
        elif op == "point":
            if n.op in OPS:
                ar, kind, _ = OPS[n.op]
                same = [o for o in g.ops if OPS[o][0] == ar and OPS[o][1] == kind]
                n.op = rng.choice(same)
            else:
                t = self._replace(t, n, g.random_leaf(rng))
        elif op == "param":
            cands = [x for x in t.nodes() if x.param is not None and x.op not in ("feat", "macro")]
            if cands:
                x = rng.choice(cands)
                if x.op == "const":
                    x.param = round(x.param * rng.choice([0.5, 0.9, 1.1, 2.0, -1.0]) + rng.gauss(0, 0.1), 4)
                else:
                    i = WINDOWS.index(x.param) if x.param in WINDOWS else 2
                    x.param = WINDOWS[min(len(WINDOWS) - 1, max(0, i + rng.choice([-1, 1])))]
            else:
                t = self._replace(t, n, g.random_leaf(rng))
        elif op == "crossover":
            donor = self._random_node(b, rng).copy()
            t = self._replace(t, n, donor)
        elif op == "hoist":
            t = n.copy() if n is not t or not t.kids else rng.choice(t.kids).copy()
        elif op == "wrap":
            name = g._pick(g.ops, rng)
            ar, kind, _ = OPS[name]
            kids = [n.copy()] + [g.random_leaf(rng) for _ in range(ar - 1)]
            rng.shuffle(kids)
            t = self._replace(t, n, Node(name, kids, rng.choice(WINDOWS) if kind == "ts" else None))
        elif op == "macro" and self.macros:
            names = [f"macro:{m}" for m in self.macros]
            mid = int(g._pick(names, rng).split(":")[1])
            t = self._replace(t, n, Node("macro", param=mid))
        return t

    # ------------------------------------------------------------- evolution
    def _tournament(self, island):
        k = island.hyper["tournament"]
        return max(island.rng.sample(island.pop, min(k, len(island.pop))), key=lambda i: i.fit)

    def init_island(self, island):
        seen, pop, tries = set(), [], 0
        # Seed with hall-of-fame models from earlier runs (warm start).
        for h in self.hof[: self.pop_size // 4]:
            ind = self.score(h.tree.copy(), island)
            if ind.key not in seen:
                seen.add(ind.key); pop.append(ind)
        while len(pop) < self.pop_size and tries < self.pop_size * 20:
            tries += 1
            d = island.rng.randint(2, island.hyper["init_depth"])
            ind = self.score(self.grammar.random_tree(island.rng, d, full=island.rng.random() < 0.5), island)
            if ind.key not in seen and ind.train > -1:
                seen.add(ind.key); pop.append(ind)
        island.pop = sorted(pop, key=lambda i: -i.fit)

    def generation(self, island):
        pop = island.pop
        new = pop[:2]  # elitism
        seen = {i.key for i in new}
        attempts = 0
        while len(new) < self.pop_size and attempts < self.pop_size * 6:
            attempts += 1
            pa, pb = self._tournament(island), self._tournament(island)
            op = island.pick_var(bool(self.macros))
            child = self.vary(op, pa.tree, pb.tree, island)
            if child.depth() > island.hyper["max_depth"] or child.size() > 60:
                continue
            fresh = child.key() not in self.score_cache
            ind = self.score(child, island)
            success = ind.fit > max(pa.fit, pb.fit if op == "crossover" else pa.fit) + 1e-6
            island.var_rate[op] = 0.95 * island.var_rate[op] + 0.05 * float(success)
            if success:
                self.grammar.reinforce(child.primitives(), amount=0.05, decay=0.999)
            # Novelty: a child identical to anything evaluated before adds no information.
            if ind.key in seen or ind.train <= -1 or not fresh:
                continue
            seen.add(ind.key)
            new.append(ind)
        for old in pop:  # top up with survivors if novelty filtering left gaps
            if len(new) >= self.pop_size:
                break
            if old.key not in seen:
                seen.add(old.key); new.append(old)
        island.pop = sorted(new, key=lambda i: -i.fit)

    def update_hof(self, size=40):
        pool = {i.key: i for i in self.hof}
        for isl in self.islands:
            for ind in isl.pop[:10]:
                pool.setdefault(ind.key, ind)
        self.hof = sorted(pool.values(), key=lambda i: -i.val)[:size]

    # ---------------------------------------------------------- meta-learning
    META_PARSIMONY = 0.001

    def meta_score(self, isl):
        """How good is an island's search? Judged on VALIDATION data with a fixed complexity
        charge on the expanded model. It must not depend on the island's own parsimony knob,
        or the meta-learner just turns the knob down and calls that progress."""
        return max(i.val - self.scale * self.META_PARSIMONY * self.expand(i.tree).size() for i in isl.pop[:5])

    def population_based_training(self):
        """Weak islands inherit (perturbed) hyperparameters and search policy from strong ones."""
        ranked = sorted(self.islands, key=lambda isl: -self.meta_score(isl))
        best, worst = ranked[0], ranked[-1]
        worst.hyper = self._perturb(best.hyper)
        worst.var_rate = dict(best.var_rate)
        # Migration: the worst island receives the best island's top models (rescored under new parsimony).
        worst.pop = sorted(worst.pop[: self.pop_size - 5] + [self.score(i.tree.copy(), worst) for i in best.pop[:5]],
                           key=lambda i: -i.fit)
        # Ring migration keeps islands from drifting apart completely.
        for a, b in zip(self.islands, self.islands[1:] + self.islands[:1]):
            b.pop[-1] = self.score(a.pop[0].tree.copy(), b)
            b.pop.sort(key=lambda i: -i.fit)

    def handle_stagnation(self, patience=2):
        """An island that stops improving keeps its elite and reseeds the rest with fresh
        random models (which now include learned macros), escaping local optima."""
        n = 0
        for isl in self.islands:
            if isl.pop[0].fit > isl.best_seen + 1e-5:
                isl.best_seen, isl.stagnant = isl.pop[0].fit, 0
                continue
            isl.stagnant += 1
            if isl.stagnant >= patience:
                keep = isl.pop[: self.pop_size // 2]
                seen = {i.key for i in keep}
                for _ in range(self.pop_size * 4):
                    if len(keep) >= self.pop_size:
                        break
                    d = isl.rng.randint(2, isl.hyper["init_depth"])
                    ind = self.score(self.grammar.random_tree(isl.rng, d, full=isl.rng.random() < 0.5), isl)
                    if ind.key not in seen and ind.train > -1:
                        seen.add(ind.key); keep.append(ind)
                isl.pop = sorted(keep, key=lambda i: -i.fit)
                isl.stagnant = 0
                n += 1
        return n

    # -------------------------------------------------------- library learning
    def learn_library(self, elite_n=30, min_count=3, max_new=2, max_lib=25):
        pool = {i.key: i for isl in self.islands for i in isl.pop}.values()
        elites = sorted(pool, key=lambda i: -(i.val - self.scale * self.META_PARSIMONY * self.expand(i.tree).size()))[:elite_n]
        counts, example = Counter(), {}
        for ind in elites:
            seen_here = set()
            for n in ind.tree.nodes():
                s = n.size()
                if n.op in OPS and 3 <= s <= 9:
                    k = n.key()
                    if k not in seen_here:
                        seen_here.add(k); counts[k] += 1; example[k] = n
        existing = {self.expand(t).key() for t in self.macros.values()}
        gain = sorted(((c * (example[k].size() - 1), k) for k, c in counts.items() if c >= min_count), reverse=True)
        added = []
        for g, k in gain:
            if len(added) >= max_new:
                break
            if len(self.macros) >= max_lib and not self._evict():
                break
            # Candidates were collected before any eviction above: skip ones that refer to a removed macro.
            if any(n.op == "macro" and n.param not in self.macros for n in example[k].nodes()):
                continue
            if self.expand(example[k]).key() in existing:
                continue
            pred = evaluate(example[k], self.task.env, self.cache, self.macros)
            if np.isfinite(pred).mean() < 0.5:
                continue
            mid = self.next_macro; self.next_macro += 1
            self.macros[mid] = example[k].copy()
            self.grammar.add_macro(mid)
            self.macro_meta[mid] = {"born_epoch": self.epoch, "count": counts[k], "task": self.task.name}
            existing.add(self.expand(example[k]).key())
            added.append(mid)
            # Compress current populations: replace occurrences with the new primitive.
            for isl in self.islands:
                isl.pop = [self.score(self._compress(i.tree, k, mid), isl) for i in isl.pop]
                isl.pop.sort(key=lambda i: -i.fit)
        self._gc_library()
        return added

    def _evict(self):
        """Drop the least-used macro that no other macro depends on."""
        self._gc_library()
        deps = {n.param for t in self.macros.values() for n in t.nodes() if n.op == "macro"}
        cands = [m for m in self.macros if m not in deps]
        if not cands:
            return False
        m = min(cands, key=lambda m: self.macro_meta.get(m, {}).get("uses", 0))
        live = [i.tree for isl in self.islands for i in isl.pop] + [i.tree for i in self.hof]
        if any(n.op == "macro" and n.param == m for t in live for n in t.nodes()):
            # Still in use: inline it so nothing breaks.
            for isl in self.islands:
                for ind in isl.pop:
                    ind.tree = self._inline(ind.tree, m); ind.key = ind.tree.key()
            for ind in self.hof:
                ind.tree = self._inline(ind.tree, m); ind.key = ind.tree.key()
        del self.macros[m]; self.grammar.remove_macro(m); self.macro_meta.pop(m, None)
        return True

    def _inline(self, tree, mid):
        if tree.op == "macro" and tree.param == mid:
            return self.macros[mid].copy()
        return Node(tree.op, [self._inline(k, mid) for k in tree.kids], tree.param)

    def _compress(self, tree, key, mid):
        if tree.key() == key:
            return Node("macro", param=mid)
        return Node(tree.op, [self._compress(k, key, mid) for k in tree.kids], tree.param)

    def _gc_library(self):
        used = Counter()
        for t in [i.tree for isl in self.islands for i in isl.pop] + [i.tree for i in self.hof] + list(self.macros.values()):
            used.update(n.param for n in t.nodes() if n.op == "macro")
        for mid in list(self.macros):
            meta = self.macro_meta.setdefault(mid, {})
            meta["uses"] = used[mid]
            if used[mid] == 0 and self.epoch - meta.get("born_epoch", 0) >= 2:
                del self.macros[mid]; self.grammar.remove_macro(mid); self.macro_meta.pop(mid)

    # -------------------------------------------------------------------- run
    def run(self, epochs=6, gens_per_epoch=5, time_budget=None):
        t0 = time.time()
        self.scale = self.estimate_scale()
        self.log(f"score scale for complexity penalties: {self.scale:.4g}")
        for isl in self.islands:
            self.init_island(isl)
        start_evals = self.evals
        for _ in range(epochs):
            self.epoch += 1
            for _ in range(gens_per_epoch):
                for isl in self.islands:
                    self.generation(isl)
            self.update_hof()
            restarts = self.handle_stagnation()
            self.population_based_training()
            added = self.learn_library()
            self.cache.trim()
            if len(self.score_cache) > 200000:
                self.score_cache.clear()
            best = self.hof[0]
            rec = {
                "task": self.task.name, "epoch": self.epoch, "evals": self.evals,
                "best_train": round(max(i.pop[0].train for i in self.islands), 5),
                "best_val": round(best.val, 5), "best_model": self.expand(best.tree).key(),
                "library": len(self.macros),
                "max_macro_level": max((self.macro_level(m) for m in self.macros), default=0),
                "new_macros": [f"M{m}={self.macros[m].key()}" for m in added if m in self.macros],
                "hyper": [isl.hyper for isl in self.islands],
                "var_rate": {k: round(np.mean([isl.var_rate[k] for isl in self.islands]), 3) for k in VAR_OPS},
                "restarts": restarts,
                "seconds": round(time.time() - t0, 1),
            }
            self.history.append(rec)
            self.log(f"epoch {self.epoch:3d} | evals {self.evals:7d} | train {rec['best_train']:+.4f} | "
                     f"val {rec['best_val']:+.4f} | library {rec['library']:2d} (depth {rec['max_macro_level']}) | "
                     f"{rec['seconds']:.0f}s" + (f" | +{', '.join(rec['new_macros'])}" if added else ""))
            if self.state_path:
                self.save(self.state_path)
            if time_budget and time.time() - t0 > time_budget:
                break
        return self.hof

    # ----------------------------------------------------------- persistence
    def save(self, path):
        state = {
            "macros": {str(k): v.to_json() for k, v in self.macros.items()},
            "macro_meta": {str(k): v for k, v in self.macro_meta.items()},
            "next_macro": self.next_macro,
            "grammar_weights": self.grammar.weights,
            "islands": [{"hyper": i.hyper, "var_rate": i.var_rate} for i in self.islands],
            "hof": {self.task.name: [[self.expand(i.tree).to_json(), i.train, i.val] for i in self.hof]},
            "history": self.history, "evals": self.evals, "epoch": self.epoch,
        }
        if os.path.exists(path):  # keep other tasks' hall of fame
            old = json.load(open(path))
            for k, v in old.get("hof", {}).items():
                state["hof"].setdefault(k, v)
        tmp = path + ".tmp"
        json.dump(state, open(tmp, "w"), indent=1)
        os.replace(tmp, path)

    def _load(self, path):
        s = json.load(open(path))
        feats = set(self.task.env)
        ok = lambda t: all(n.op != "feat" or n.param in feats for n in t.nodes()) and \
            all(n.op not in OPS or n.op in self.grammar.ops for n in t.nodes())
        # Library: keep macros whose features/ops exist in this task (dependencies first).
        for k in sorted(s.get("macros", {}), key=int):
            t = Node.from_json(s["macros"][k])
            if ok(t) and all(n.param in self.macros for n in t.nodes() if n.op == "macro"):
                self.macros[int(k)] = t
                self.grammar.add_macro(int(k))
                self.macro_meta[int(k)] = s.get("macro_meta", {}).get(k, {})
                self.macro_meta[int(k)]["born_epoch"] = 0
        self.next_macro = s.get("next_macro", 0)
        # Experience is a prior, not a rule: blend learned preferences halfway back to
        # uniform so primitives the past never needed (e.g. sin) remain reachable.
        same_task = self.task.name in s.get("hof", {})
        for name, w in s.get("grammar_weights", {}).items():
            if name in self.grammar.weights or (name.startswith("macro:") and int(name[6:]) in self.macros):
                self.grammar.weights[name] = w if same_task else 0.5 * w + 0.5
        for isl, saved in zip(self.islands, s.get("islands", [])):
            isl.hyper, isl.var_rate = saved["hyper"], saved["var_rate"]
        for j, tr, va in s.get("hof", {}).get(self.task.name, []):
            t = Node.from_json(j)
            if ok(t):
                self.hof.append(Individual(t, tr, va))
        self.history = s.get("history", [])
        self.evals = s.get("evals", 0)
        self.epoch = s.get("epoch", 0)
        self.log(f"loaded state: {len(self.macros)} macros, {len(self.hof)} hall-of-fame models, "
                 f"{len(self.history)} past epochs")
