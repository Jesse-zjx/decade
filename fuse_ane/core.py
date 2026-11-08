"""FUSE-ANE reference numerical core. No model calls or candidate execution.

This is an independently written experimental heuristic, not a reproduction
of ALNS/AdaEdit and not an implementation of their theoretical guarantees.
The execution plan defines collection, provenance, budgets and valid actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Mapping, Sequence
import hashlib
import math
import random


@dataclass(frozen=True)
class Action:
    action_id: str
    parent_sha: str
    region_sha: str
    scope: str  # B, C, G
    operation: str = "rewrite"

    @property
    def key(self) -> str:
        return "|".join((self.parent_sha, self.region_sha, self.scope, self.operation))


@dataclass(frozen=True)
class Outcome:
    key: str
    scope: str
    stalled_before: bool
    reward: float
    cost: float


def progress_reward(parent: float, incumbent: float, child: float | None,
                    scale: float, *, complete_valid: bool,
                    new_source: bool) -> tuple[float, float | None, float | None]:
    """Return operational reward and unmodified signed gains (percentage points).

    Missing candidate quality is not coerced to a numerical quality. Failures
    have zero realized operational reward, a separate variable.
    """
    if not all(math.isfinite(x) and x > 0 for x in (parent, incumbent, scale)):
        raise ValueError("Parent, incumbent and reference scale must be positive/finite")
    if not complete_valid:
        return 0.0, None, None
    if child is None or not math.isfinite(child) or child <= 0:
        raise ValueError("Valid outcomes require positive finite child quality")
    gp = 100.0 * (parent - child) / scale
    ge = 100.0 * (incumbent - child) / scale
    if not new_source:
        return 0.0, gp, ge
    clip = lambda x: min(2.0, max(0.0, x)) / 2.0
    return 0.7 * clip(ge) + 0.3 * clip(gp), gp, ge


def resource_cost(tokens: int, started_instances: int) -> float:
    if tokens < 0 or started_instances < 0:
        raise ValueError("Resource consumption cannot be negative")
    if tokens == 0 and started_instances == 0:
        raise ValueError("An executed generation episode cannot be cost-free")
    return tokens / 80000.0 + started_instances / 10240.0


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:16], "big")


class CreditScheduler:
    def __init__(self, seed: int, mode: str = "F"):
        if mode not in {"F", "U", "A"}:
            raise ValueError("mode must be F (cost-aware), U (uniform), A (cost-blind)")
        self.rng = random.Random(seed)
        self.seed = seed
        self.mode = mode
        self.history: list[Outcome] = []
        self.completed = 0

    def update(self, action: Action, stalled_before: bool,
               reward: float, tokens: int, started_instances: int) -> None:
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Reward must be finite in [0,1]")
        self.history.append(Outcome(action.key, action.scope, stalled_before,
                                    reward, resource_cost(tokens, started_instances)))
        self.history = self.history[-32:]
        self.completed += 1

    def scores(self, actions: Sequence[Action], stalled: bool) -> dict[str, dict[str, float]]:
        if not actions or len({a.action_id for a in actions}) != len(actions):
            raise ValueError("The menu must be nonempty with unique action IDs")
        records = [(o, 0.95 ** age) for age, o in enumerate(reversed(self.history))]
        nall = sum(w for _, w in records)
        result = {}
        for a in actions:
            if a.scope not in {"B", "C", "G"}:
                raise ValueError("Unknown scope")
            pool = [(o, w) for o, w in records
                    if o.scope == a.scope and o.stalled_before == stalled]
            local = [(o, w) for o, w in records if o.key == a.key]
            npool = sum(w for _, w in pool)
            nlocal = sum(w for _, w in local)
            mup = (sum(w * o.reward for o, w in pool) + 2 * .10) / (npool + 2)
            cp = (sum(w * o.cost for o, w in pool) + 2 * .0375) / (npool + 2)
            mu = (sum(w * o.reward for o, w in local) + 2 * mup) / (nlocal + 2)
            cost = (sum(w * o.cost for o, w in local) + 2 * cp) / (nlocal + 2)
            bonus = math.sqrt(.1 * math.log(2 + nall) / (1 + nlocal + npool / 4))
            numerator = mu + bonus
            score = numerator if self.mode == "A" else numerator / max(cost, 1e-4)
            result[a.action_id] = {"mean_reward": mu, "mean_cost": cost,
                                   "bonus": bonus, "score": score,
                                   "local_weight": nlocal, "pool_weight": npool}
        return result

    def _uniform(self, actions: Sequence[Action], required_scope: str | None = None) -> Action:
        scopes = sorted({a.scope for a in actions})
        scope = required_scope if required_scope in scopes else self.rng.choice(scopes)
        if required_scope and required_scope not in scopes and "G" in scopes:
            scope = "G"
        scoped = [a for a in actions if a.scope == scope]
        parent = self.rng.choice(sorted({a.parent_sha for a in scoped}))
        return self.rng.choice(sorted((a for a in scoped if a.parent_sha == parent),
                                     key=lambda a: a.action_id))

    def choose(self, actions: Sequence[Action], stalled: bool) -> tuple[Action, str, Mapping]:
        scores = self.scores(actions, stalled)
        episode = self.completed + 1
        if episode <= 6:
            desired = ("B", "C", "G", "B", "G", "C")[episode - 1]
            return self._uniform(actions, desired), "warmup", scores
        if episode % 5 == 0:
            return self._uniform(actions, "G"), "global_exploration", scores
        if self.mode == "U" or self.rng.random() < .20:
            return self._uniform(actions), "uniform_exploration", scores
        selected = min(actions, key=lambda a: (-scores[a.action_id]["score"],
                          stable_seed(self.seed, episode, a.action_id)))
        return selected, "learned", scores


def bootstrap_quality(task_pairs: Mapping[str, Sequence[tuple[float, float]]],
                      repeats: int = 50000, seed: int = 2026100201) -> dict:
    """Paired search-group bootstrap, task-equal macro average. Requires NumPy.

    Each tuple is (method loss, HiFo loss) of a preselected complete program.
    Missing groups raise rather than being silently dropped.
    """
    import numpy as np
    if not task_pairs or repeats < 100:
        raise ValueError("Need tasks and at least 100 bootstrap resamples")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = []
    means = []
    task_results = {}
    for task, pairs in sorted(task_pairs.items()):
        a = np.asarray(pairs, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2:
            raise ValueError("Each task requires at least two paired groups")
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ValueError("Missing or invalid group quality; do not drop the group")
        d = 100 * (a[:, 0] - a[:, 1]) / a[:, 1]
        sample = d[rng.integers(0, len(d), (repeats, len(d)))].mean(axis=1)
        means.append(float(d.mean()))
        draws.append(sample)
        task_results[task] = {"groups": len(a), "mean_D_pp": float(d.mean()),
                              "two_sided_95": np.quantile(sample, [.025, .975]).tolist()}
    macro = np.stack(draws).mean(axis=0)
    return {"tasks": task_results, "macro_D_pp": sum(means) / len(means),
            "two_sided_95": np.quantile(macro, [.025, .975]).tolist(),
            "one_sided_95_upper": float(np.quantile(macro, .95)),
            "note": "Conditional on frozen programs/settings; independent search groups are units"}
