"""Frozen v1 action controller; no model requests or candidate execution.

Canonical programs are ``{hash, source, mean, factors}``, where ``mean`` is
the completed Search mean loss (lower is better). Factors use ``source``,
``start`` (UTF-8 byte offset), ``factor_version``, ``background`` and
``local_param_eligible``. Credit records are separate, keyed by
``credit_key(program_hash, factor)``, with ``mean`` and nullable ``se``.

The caller owns background/panel validation, archive/reservoir, actual send
visits, completion usage, and saved predecision rows. It must supply only
information available before the request. This module cannot infer that a
request was sent from a selection or that a missing quality means zero.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
from typing import Any, Mapping, Sequence

import numpy as np


OPS = ("LOCAL_REWRITE", "LOCAL_PARAM", "MACRO_M1", "MACRO_M2", "MACRO_E1", "MACRO_E2")
Z_NAMES = (
    "intercept", *("op_" + op for op in OPS), "gap_pp_div5", "parent_rank",
    "parent_visits", "target_visits", "target_bytes", "payload_bytes",
    "main_progress", "helper_count", "operation_valid_rate",
)
CREDIT_NAMES = ("credit_mean", "mean_abs_credit", "credit_se", "credit_available", "se_available")
U_NAMES = tuple(op + ":" + name for op in OPS for name in CREDIT_NAMES)
MODEL_VERSION = "fuse-credit-v1-ridge-z16-u30-lambda1-beta025-window512"
TIE_ATOL = 1e-12


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def derive_seed(seed: int, *parts: Any) -> int:
    """Stable independent substreams; never Python's randomized hash()."""
    return int.from_bytes(hashlib.sha256(_json([int(seed), *parts]).encode()).digest()[:16], "big")


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _factor_version(factor: Mapping[str, Any]) -> str:
    value = factor.get("factor_version", factor.get("version"))
    if not value:
        raise ValueError("factor_version is required")
    return str(value)


def credit_key(program_hash: str, factor: Mapping[str, Any]) -> str:
    """Identity includes the complete program and the current background."""
    return _json([program_hash, _factor_version(factor), factor["background"]])


def action_sort_key(action: Mapping[str, Any]) -> tuple:
    target = action.get("target")
    parents = tuple(action["parents"])
    return (parents[0], OPS.index(action["op"]), int(target["start"]) if target else -1,
            parents, _factor_version(target) if target else "")


def action_identity(action: Mapping[str, Any]) -> str:
    target = action.get("target")
    return _json([action["op"], list(action["parents"]),
                  credit_key(action["parents"][0], target) if target else None])


def build_slate(programs: Sequence[Mapping[str, Any]], partner_seed: int,
                *, return_audit: bool = False) -> Any:
    """Enumerate actual local/macro actions; partners frozen before scoring.

    EoH uses rank weights 1/(rank+1+N) with replacement. Each eligible anchor
    receives one independently drawn partner for E1 and E2. Identical anchor
    and partner hashes remain possible, as in the native selection rule.
    ``programs`` must already be the active 8-elite + 4-reservoir pool.
    """
    by_hash = {p["hash"]: p for p in programs}
    if len(by_hash) != len(programs):
        raise ValueError("active programs must be deduplicated by complete hash")
    ranked = sorted(programs, key=lambda p: (_finite(p["mean"], "mean"), p["hash"]))
    if not ranked:
        raise ValueError("no complete program available for action domain")
    weights = [1.0 / (rank + 1 + len(ranked)) for rank in range(len(ranked))]
    actions, partner_draws = [], []
    rng = random.Random(int(partner_seed))
    for program in sorted(programs, key=lambda p: p["hash"]):
        parent = program["hash"]
        for factor in sorted(program["factors"], key=lambda f: (f["start"], _factor_version(f))):
            if not factor.get("local_eligible", True):
                continue
            actions.append({"op": OPS[0], "parents": [parent], "target": copy.deepcopy(factor)})
            if factor.get("local_param_eligible", False):
                actions.append({"op": OPS[1], "parents": [parent], "target": copy.deepcopy(factor)})
        for op in OPS[2:]:
            parents = [parent]
            if op in OPS[4:]:
                partner = rng.choices(ranked, weights=weights, k=1)[0]["hash"]
                parents.append(partner)
                partner_draws.append({"anchor": parent, "op": op, "partner": partner})
            actions.append({"op": op, "parents": parents, "target": None})
    actions.sort(key=action_sort_key)
    audit = {"seed": int(partner_seed), "ranked_program_hashes": [p["hash"] for p in ranked],
             "rank_weights": weights, "with_replacement": True, "partner_draws": partner_draws}
    return (actions, audit) if return_audit else actions


def policy_credit_maps(programs: Sequence[Mapping[str, Any]], seed: int,
                       credits: Mapping[str, Any] | None = None) -> dict:
    """Preserve true null slots and derange only currently known records.

    A single map is reused by all local operations and macro aggregations
    within a decision. Records must already be current-background validated.
    S is a seeded uniform random permutation conditioned on no fixed point.
    """
    credits = credits or {}
    current = {}
    for program in sorted(programs, key=lambda p: p["hash"]):
        for factor in sorted(program["factors"], key=lambda f: (f["start"], _factor_version(f))):
            key = credit_key(program["hash"], factor)
            record = credits.get(key)
            if record is None or record.get("mean") is None:
                current[key] = None
            else:
                _finite(record["mean"], "credit mean")
                if record.get("se") is not None:
                    _nonnegative(record["se"], "credit se")
                current[key] = copy.deepcopy(record)
    available = sorted(k for k, record in current.items() if record is not None)
    shuffled = copy.deepcopy(current)
    perm = list(range(len(available)))
    rng = random.Random(int(seed))
    if len(available) > 1:
        while True:
            rng.shuffle(perm)
            if all(i != j for i, j in enumerate(perm)):
                break
    mapping = {}
    for destination, source_index in zip(available, perm):
        source = available[source_index]
        shuffled[destination] = copy.deepcopy(current[source])
        mapping[destination] = source
    return {"C": current, "S": shuffled,
            "audit": {"seed": int(seed), "available_count": len(available),
                      "permutation_identity": len(available) <= 1,
                      "mapping": mapping, "null_keys": sorted(k for k in current if current[k] is None)}}


def _raw_credit(action: Mapping[str, Any], programs: Mapping[str, Mapping[str, Any]],
                credits: Mapping[str, Any]) -> dict:
    if action["target"] is not None:
        slots = [(action["parents"][0], action["target"])]
    else:
        slots = [(parent, factor) for parent in action["parents"] for factor in programs[parent]["factors"]]
    keys = [credit_key(parent, factor) for parent, factor in slots]
    records = [credits.get(key) for key in keys]
    means = [_finite(r["mean"], "credit mean") for r in records if r is not None and r.get("mean") is not None]
    ses = [_nonnegative(r["se"], "credit se") for r in records
           if r is not None and r.get("mean") is not None and r.get("se") is not None]
    return {"keys": keys, "records": copy.deepcopy(records), "factor_count": len(slots),
            "mean": statistics.mean(means) if means else None,
            "mean_abs": statistics.mean(abs(c) for c in means) if means else None,
            "se": statistics.mean(ses) if ses else None,
            "available": len(means) / len(slots) if slots else 0.0,
            "se_available": len(ses) / len(slots) if slots else 0.0}


def make_features(action: Mapping[str, Any], programs_by_hash: Mapping[str, Mapping[str, Any]],
                  *, metadata: Mapping[str, Any], credits: Mapping[str, Any] | None = None,
                  arm: str = "C") -> dict:
    """Create one JSON-safe row from completed-state information.

    Required metadata: ``scale`` (frozen positive reference s),
    ``payload_utf8_bytes`` (all actually rendered messages), ``progress``.
    Optional metadata: ``elite_mean``; ``ranks`` (zero-based hash->rank);
    ``active_count``; ``parent_visits`` (hash->sent count); ``target_visits``
    (credit_key->sent count); ``op_history`` (op->{attempts,valid}); and
    ``completion_history`` (op->[known completion tokens or None]). Counts
    must be the run's actually sent actions, never CF evaluation counts.
    Explicit scalar gap/rank/visit/helper overrides are also supported.
    """
    if arm not in ("C", "U", "S", "R", "G"):
        raise ValueError("unknown controller arm")
    op = action["op"]
    op_index = OPS.index(op)
    parents = [programs_by_hash[p] for p in action["parents"]]
    if len(parents) != (2 if op in OPS[4:] else 1):
        raise ValueError("action has incorrect number of actual parents")
    target = action.get("target")
    if (target is not None) != op.startswith("LOCAL_"):
        raise ValueError("local actions require exactly one actual target")
    scale = _finite(metadata["scale"], "scale")
    if scale <= 0:
        raise ValueError("frozen reference scale must be positive")
    ranked = sorted(programs_by_hash.values(), key=lambda p: (p["mean"], p["hash"]))
    ranks = metadata.get("ranks", {p["hash"]: i for i, p in enumerate(ranked)})
    best_parent = min(parents, key=lambda p: (p["mean"], ranks[p["hash"]], p["hash"]))
    elite_mean = _finite(metadata.get("elite_mean", ranked[0]["mean"]), "elite_mean")
    gap = _finite(metadata.get("gap_pp", 100 * (best_parent["mean"] - elite_mean) / scale), "gap_pp")
    active_count = int(metadata.get("active_count", len(programs_by_hash)))
    parent_rank = _nonnegative(metadata.get("parent_rank", ranks[best_parent["hash"]]), "parent_rank")
    parent_counts = metadata.get("parent_visits", {})
    parent_visits = (statistics.mean(parent_counts.get(p["hash"], 0) for p in parents)
                     if isinstance(parent_counts, Mapping) else parent_counts)
    parent_visits = _nonnegative(parent_visits, "parent_visits")
    target_counts = metadata.get("target_visits", {})
    target_visits = (target_counts.get(credit_key(parents[0]["hash"], target), 0)
                     if target and isinstance(target_counts, Mapping) else target_counts if target else 0)
    target_visits = _nonnegative(target_visits, "target_visits")
    target_bytes = len(target["source"].encode("utf-8")) if target else 0
    payload_bytes = _nonnegative(metadata["payload_utf8_bytes"], "payload_utf8_bytes")
    progress = _finite(metadata["progress"], "progress")
    if not 0 <= progress <= 1:
        raise ValueError("main position progress must be within [0,1]")
    helper_count = _nonnegative(metadata.get("helper_count", sum(len(p["factors"]) for p in parents)), "helper_count")
    op_history = metadata.get("op_history", {}).get(op, {})
    attempts = _nonnegative(op_history.get("attempts", 0), "operation attempts")
    valid = _nonnegative(op_history.get("valid", 0), "operation valid count")
    if valid > attempts:
        raise ValueError("valid count cannot exceed attempts")
    valid_rate = valid / attempts if attempts else 0.5
    z = [1.0, *[float(i == op_index) for i in range(6)],
         min(max(gap / 5, 0), 1), parent_rank / max(1, active_count - 1),
         min(math.log1p(parent_visits) / math.log(513), 1),
         min(math.log1p(target_visits) / math.log(513), 1),
         min(target_bytes / 32768, 1), min(payload_bytes / 32768, 1), progress,
         min(helper_count / 16, 1), valid_rate]
    raw_credit = _raw_credit(action, programs_by_hash, credits or {})
    base = [max(-1, min(raw_credit["mean"] / 5, 1)) if raw_credit["mean"] is not None else 0.0,
            min(raw_credit["mean_abs"] / 5, 1) if raw_credit["mean_abs"] is not None else 0.0,
            min(raw_credit["se"] / 5, 1) if raw_credit["se"] is not None else 0.0,
            raw_credit["available"], raw_credit["se_available"]]
    raw_u = [v if i == op_index else 0.0 for i in range(6) for v in base]
    u = [0.0] * 30 if arm == "U" else raw_u
    input_tokens = math.ceil(payload_bytes / 4)
    known = [_nonnegative(v, "completion tokens") for v in metadata.get("completion_history", {}).get(op, []) if v is not None][-32:]
    expected_response_bytes = target_bytes if target else max(len(p["source"].encode("utf-8")) for p in parents)
    output_tokens = statistics.median(known) if known else math.ceil(expected_response_bytes / 4) + 512
    row = {"action": copy.deepcopy(action), "action_id": action_identity(action), "arm": arm,
           "z": z, "u": u, "x": z + u, "raw_u": raw_u,
           "raw": {"gap_pp": gap, "parent_rank": parent_rank, "active_count": active_count,
                   "parent_visits": parent_visits, "target_visits": target_visits,
                   "target_utf8_bytes": target_bytes, "payload_utf8_bytes": payload_bytes,
                   "progress": progress, "helper_count": helper_count,
                   "op_attempts": attempts, "op_valid": valid, "credit": raw_credit},
           "gap_to_elite_pp": gap, "predicted_input_tokens": input_tokens,
           "predicted_output_tokens": output_tokens, "predicted_tokens": input_tokens + output_tokens,
           "token_prediction": {"completion_samples": len(known), "expected_response_utf8_bytes": expected_response_bytes,
                                "output_rule": "median_known_last32" if known else "ceil_bytes_div4_plus512"}}
    if len(z) != 16 or len(u) != 30:
        raise AssertionError("frozen feature dimensions changed")
    _json(row)
    return row


def mask_feature_row(row: Mapping[str, Any]) -> dict:
    """Mask stored predecision features; never reconstruct from future credit."""
    result = copy.deepcopy(row)
    result.update(arm="U", u=[0.0] * 30, x=list(row["z"]) + [0.0] * 30)
    return result


def _uniform_index(count: int, seed: int) -> tuple[int, float]:
    if count <= 0:
        raise ValueError("cannot choose from empty domain")
    draw = random.Random(int(seed)).random()
    return min(count - 1, int(draw * count)), draw


def stratified_uniform(actions: Sequence[Mapping[str, Any]], seed: int) -> tuple[int, dict]:
    """Uniform family -> operation -> anchor parent -> target choice."""
    indices = sorted(range(len(actions)), key=lambda i: action_sort_key(actions[i]))
    trace = {}
    partitions = (
        ("family", lambda a: "local" if a["op"].startswith("LOCAL_") else "macro"),
        ("operator", lambda a: OPS.index(a["op"])),
        ("parent", lambda a: a["parents"][0]),
        ("target", lambda a: (int(a["target"]["start"]), _factor_version(a["target"])) if a["target"] else (-1, "")),
    )
    for name, key in partitions:
        choices = sorted({key(actions[i]) for i in indices})
        sub_seed = derive_seed(seed, name)
        selected, draw = _uniform_index(len(choices), sub_seed)
        chosen = choices[selected]
        indices = [i for i in indices if key(actions[i]) == chosen]
        trace[name] = {"choices": choices, "selected": chosen, "draw": draw, "seed": sub_seed}
    if len(indices) != 1:
        raise ValueError("duplicate action identity within hierarchical domain")
    return indices[0], trace


class RidgeController:
    """46-feature ridge with a separate 16-feature public exploration model."""

    def __init__(self, arm: str = "C") -> None:
        if arm not in ("C", "U", "S", "R", "G"):
            raise ValueError("unknown controller arm")
        self.arm = arm
        self.history: list[dict] = []
        self.total_completed = 0

    def _policy_row(self, row: Mapping[str, Any]) -> dict:
        if self.arm == "U":
            return mask_feature_row(row)
        if self.arm == "S" and row.get("arm") != "S":
            raise ValueError("S requires saved features from its current predecision permutation")
        return copy.deepcopy(row)

    def observe(self, saved_row: Mapping[str, Any], reward_pp: float, valid: bool,
                actual_cost: Any = None, **outcome_metadata: Any) -> None:
        """Append only a completed action's saved predecision model features.

        For a failed candidate the caller supplies actual retain-parent reward
        zero; its candidate quality remains null in outcome metadata. A merely
        pending request must not call this method.
        """
        reward = _finite(reward_pp, "operational reward")
        if not valid and reward != 0:
            raise ValueError("failed candidate retain-parent operational reward must be zero")
        row = self._policy_row(saved_row)
        if len(row["z"]) != 16 or len(row["x"]) != 46:
            raise ValueError("invalid frozen feature dimensions")
        item = {"row": row, "reward_pp": reward, "label": max(-1.0, min(reward / 5, 1.0)),
                "valid": bool(valid), "actual_cost": copy.deepcopy(actual_cost),
                "outcome": copy.deepcopy(outcome_metadata)}
        _json(item)
        self.history.append(item)
        self.history = self.history[-512:]
        self.total_completed += 1

    def fit(self) -> dict:
        history = self.history[-512:]
        if history:
            x = np.asarray([r["row"]["x"] for r in history], dtype=float)
            z = np.asarray([r["row"]["z"] for r in history], dtype=float)
            y = np.asarray([r["label"] for r in history], dtype=float)
            a = np.eye(46) + x.T @ x
            theta = np.linalg.solve(a, x.T @ y)
            az = np.eye(16) + z.T @ z
        else:
            theta, az = np.zeros(46), np.eye(16)
        return {"theta": theta, "Az": az, "n": len(history)}

    def choose(self, feature_rows: Sequence[Mapping[str, Any]], *, main_position: int,
               random_seed: int, force_uniform: bool = False) -> dict:
        """Choose before successor generation; main_position is one-based.

        main_position counts non-initialization positions from the start of
        the run; fork step 1 therefore uses 17, never a reset to 1. Shared
        random_seed replays exploration coin, strata draws, and tie quantile.
        """
        if main_position < 1:
            raise ValueError("main_position must be one-based")
        if not feature_rows:
            raise ValueError("empty candidate feature slate")
        indexed = sorted(enumerate(feature_rows), key=lambda pair: action_sort_key(pair[1]["action"]))
        rows = [self._policy_row(r) for _, r in indexed]
        ids = [r["action_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate action identities")
        fit = self.fit()
        x = np.asarray([r["x"] for r in rows], dtype=float)
        z = np.asarray([r["z"] for r in rows], dtype=float)
        means = 5 * (x @ fit["theta"])
        variance = np.einsum("ij,ji->i", z, np.linalg.solve(fit["Az"], z.T))
        bonuses = 5 * 0.25 * np.sqrt(np.maximum(variance, 0))
        scores = []
        for i, row in enumerate(rows):
            positive_gain = max(0.0, float(means[i]) - row["gap_to_elite_pp"])
            denominator = max(row["predicted_tokens"] / 1000, 0.25)
            score = (positive_gain + float(bonuses[i])) / denominator
            scores.append(score)
            row["prediction"] = {"mu_pp": float(means[i]), "public_bonus_pp": float(bonuses[i]),
                                 "positive_gain_pp": positive_gain, "token_denominator": denominator, "score": score}
        coin_seed = derive_seed(random_seed, "exploration_coin")
        coin = random.Random(coin_seed).random()
        cold_start = main_position <= 16
        explore = bool(force_uniform or self.arm == "R" or cold_start or coin < 0.2)
        exploration_trace = None
        tie_indices = []
        tie_seed = derive_seed(random_seed, "tie_break")
        _, tie_draw = _uniform_index(1, tie_seed)
        if explore:
            selected, exploration_trace = stratified_uniform([r["action"] for r in rows], derive_seed(random_seed, "strata"))
        else:
            maximum = max(scores)
            tie_indices = [i for i, score in enumerate(scores) if abs(score - maximum) <= TIE_ATOL]
            tie_position, tie_draw = _uniform_index(len(tie_indices), tie_seed)
            selected = tie_indices[tie_position]
        return {"model_version": MODEL_VERSION, "arm": self.arm, "main_position": int(main_position),
                "random_seed": int(random_seed), "exploration_coin": coin, "exploration_coin_seed": coin_seed,
                "cold_start": cold_start, "force_uniform": bool(force_uniform), "exploration": explore,
                "exploration_trace": exploration_trace, "tie_indices": tie_indices,
                "tie_seed": tie_seed, "tie_draw": tie_draw, "tie_atol": TIE_ATOL,
                "training_rows": fit["n"], "total_completed": self.total_completed,
                "theta": fit["theta"].tolist(), "Az": fit["Az"].tolist(),
                "candidate_rows": rows, "selected_index": selected,
                "selected_input_index": indexed[selected][0], "selected_row": rows[selected],
                "selected_action": rows[selected]["action"],
                "slate_sha256": hashlib.sha256(_json(rows).encode()).hexdigest()}

    def shadow_without_credit(self) -> "RidgeController":
        """Fit U from C's own actual historical actions and saved public rows."""
        model = RidgeController("U")
        model.history = copy.deepcopy(self.history)
        for item in model.history:
            item["row"] = mask_feature_row(item["row"])
        model.total_completed = self.total_completed
        return model

    def to_dict(self) -> dict:
        value = {"model_version": MODEL_VERSION, "arm": self.arm,
                 "history": copy.deepcopy(self.history), "total_completed": self.total_completed}
        _json(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RidgeController":
        if value["model_version"] != MODEL_VERSION:
            raise ValueError("model version differs from frozen v1 specification")
        result = cls(value["arm"])
        result.history = copy.deepcopy(value["history"])
        if len(result.history) > 512:
            raise ValueError("saved model exceeds recent-512 history")
        result.total_completed = int(value["total_completed"])
        for item in result.history:
            if len(item["row"]["z"]) != 16 or len(item["row"]["x"]) != 46:
                raise ValueError("saved feature dimensions differ")
            if result.arm == "U" and any(item["row"]["x"][16:]):
                raise ValueError("U saved state contains forbidden credit features")
            if result.arm == "S" and item["row"].get("arm") != "S":
                raise ValueError("S saved history lacks predecision permuted features")
        _json(result.to_dict())
        return result
