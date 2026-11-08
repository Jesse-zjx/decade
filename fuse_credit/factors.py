"""Bounded, non-executing function registry for the frozen v1 local domain.

Offsets are UTF-8 byte offsets, never character positions. This module does not
execute candidates or establish quality/purity. Unsupported bindings remove a
function from the local domain, not the complete module from macro evolution.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import json
import math
import symtable

ENTRY_NAME = "select_next_node"
REGISTRY_VERSION = "bounded-top-level-functions-v1"
_DYNAMIC = {"eval", "exec", "compile", "globals", "locals", "vars", "getattr",
            "setattr", "delattr", "__import__"}
_BUILTINS = set(dir(builtins))


class ContractError(ValueError):
    """A response or stale target violates the frozen local contract."""


def sha(source):
    data = source.encode("utf-8") if isinstance(source, str) else source
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False))


def _span(node, lines):
    # A full final physical line includes its comment and line ending. This
    # keeps ordinary fenced-function replacements exactly reversible.
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno]))
    return start, end


def _signature(node):
    return json.dumps({"name": node.name,
                       "args": ast.dump(node.args, include_attributes=False),
                       "returns": ast.dump(node.returns, include_attributes=False)
                       if node.returns is not None else None,
                       "type_comment": node.type_comment}, sort_keys=True)


def _finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _body_param_shape(node):
    result = copy.deepcopy(node)
    for statement in result.body:
        for item in ast.walk(statement):
            if isinstance(item, ast.Constant) and _finite_number(item.value):
                item.value = 0
                item.kind = None
    return ast.dump(result, include_attributes=False)


def _global_references(table):
    names = {s.get_name() for s in table.get_symbols()
             if s.is_global() and s.is_referenced()}
    for child in table.get_children():
        names.update(_global_references(child))
    return names


def _target_names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)}


def registry(source, context_binding):
    """Return eligible factors and explicit static exclusions, without exec.

    ``context_binding`` must identify evaluator, panel, execution seeds/runtime
    (the caller freezes these). The worker's documented injected ``np`` binding
    is included even when the generated module does not import NumPy itself.
    """
    if not isinstance(source, str) or not isinstance(context_binding, dict) or not context_binding:
        raise ContractError("SOURCE_AND_NONEMPTY_EXECUTION_BINDING_REQUIRED")
    tree = ast.parse(source)
    data = source.encode("utf-8")
    lines = data.splitlines(keepends=True)
    table = symtable.symtable(source, "<candidate-static>", "exec")
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    declarations = {}
    global_dynamic = False
    for n in tree.body:
        start, end = _span(n, lines)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names, kind = {n.name}, "function" if isinstance(n, ast.FunctionDef) else "unsupported"
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names = {a.asname or (a.name.split(".")[0] if isinstance(n, ast.Import) else a.name)
                     for a in n.names}
            kind = "import"
            global_dynamic |= "*" in names
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            names, kind = _target_names(n), "global_value"
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            continue  # Module docstring.
        else:
            # Conditional/import-time rebinding has no unique static slot.
            global_dynamic = True
            continue
        for name in names:
            declarations.setdefault(name, []).append({"kind": kind, "source_hash": sha(data[start:end]),
                                                      "binding_ordinal": len(declarations.get(name, []))})
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _DYNAMIC:
            global_dynamic = True
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            # No alias/rebinding inference is claimed for mutable namespaces.
            global_dynamic = True
    by_name = {n.name: n for n in functions}
    reasons, references, calls = {}, {}, {}
    for n in functions:
        why = []
        if n.name == ENTRY_NAME:
            why.append("ENTRY_FUNCTION")
        if global_dynamic:
            why.append("UNRESOLVED_DYNAMIC_MODULE_BINDING")
        if len(declarations.get(n.name, [])) != 1:
            why.append("NONUNIQUE_FUNCTION_BINDING")
        if isinstance(n, ast.AsyncFunctionDef) or n.decorator_list or getattr(n, "type_params", []):
            why.append("ASYNC_DECORATED_OR_GENERIC_FUNCTION")
        if any(isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                  ast.Lambda, ast.Yield, ast.YieldFrom, ast.Await))
               for stmt in n.body for child in ast.walk(stmt)):
            why.append("CLOSURE_OR_GENERATOR")
        matching = [t for t in table.get_children() if t.get_name() == n.name and t.get_lineno() == n.lineno]
        refs = _global_references(matching[0]) if len(matching) == 1 else set()
        # Defaults and annotations are resolved at definition time too.
        for part in [n.args, n.returns]:
            if part is not None:
                refs.update(x.id for x in ast.walk(part)
                            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load))
        manifest = {}
        for name in sorted(refs):
            slots = declarations.get(name, [])
            if len(slots) == 1 and slots[0]["kind"] != "unsupported":
                manifest[name] = slots[0]
            elif not slots and name in _BUILTINS:
                manifest[name] = {"kind": "builtin", "name": name}
            elif not slots and name == "np":
                manifest[name] = {"kind": "worker_injected_numpy", "name": "np"}
            else:
                why.append("UNRESOLVED_REFERENCE:" + name)
        # Statically named helper dependencies are the only inter-function
        # dependencies supported. Imported calls and ordinary array methods
        # remain allowed; no purity assertion follows from this check.
        local_imports = {a.asname or a.name.split(".")[0]
                         for item in ast.walk(n) if isinstance(item, (ast.Import, ast.ImportFrom))
                         for a in item.names}
        for item in ast.walk(n):
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                called = item.func.id
                if called not in refs and called not in local_imports:
                    why.append("DYNAMIC_LOCAL_CALL:" + called)
                elif called in manifest and manifest[called]["kind"] == "global_value":
                    why.append("DYNAMIC_GLOBAL_CALL:" + called)
        references[n.name] = manifest
        calls[n.name] = {name for name in refs if name in by_name}
        reasons[n.name] = why
    def recursive(name, trail):
        if name in trail:
            return True
        return any(recursive(dep, trail | {name}) for dep in calls.get(name, set()))
    for n in functions:
        if recursive(n.name, set()):
            reasons[n.name].append("RECURSIVE_DEPENDENCY")
    # A target depending on an unsupported helper cannot claim resolved local
    # bindings. ENTRY_FUNCTION itself is not a dependency defect.
    changed = True
    while changed:
        changed = False
        for name, deps in calls.items():
            if not reasons[name] and any(any(r != "ENTRY_FUNCTION" for r in reasons[d]) for d in deps):
                reasons[name].append("UNSUPPORTED_HELPER_DEPENDENCY")
                changed = True
    entry_bindings = {name: slots for name, slots in sorted(declarations.items())}
    result, excluded = [], []
    for n in functions:
        start, end = _span(n, lines)
        if reasons[n.name]:
            excluded.append({"name": n.name, "start": start, "reasons": sorted(set(reasons[n.name]))})
            continue
        signature = _signature(n)
        occurrence = {"name": n.name, "binding_ordinal": 0}
        manifest = references[n.name]
        function_source = data[start:end].decode("utf-8")
        version = _digest({"source": function_source, "signature": signature,
                           "occurrence": occurrence, "bindings": manifest})
        # Target implementation bytes are deliberately absent from background;
        # the exact surrounding bytes and all other bindings remain present.
        other_bindings = copy.deepcopy(entry_bindings)
        other_bindings[n.name] = [{"kind": "target_function", "signature": signature}]
        background = _digest({"prefix": data[:start].decode("utf-8"),
                              "suffix": data[end:].decode("utf-8"),
                              "entry_and_bindings": other_bindings,
                              "execution": context_binding})
        result.append({"name": n.name, "start": start, "end": end, "source": function_source,
                       "signature": signature, "occurrence": occurrence, "version": version,
                       "factor_version": version, "background": background,
                       "program_hash": sha(source), "binding_manifest": manifest,
                       "context_binding": copy.deepcopy(context_binding),
                       "local_param_eligible": any(isinstance(v, ast.Constant) and _finite_number(v.value)
                                                   for stmt in n.body for v in ast.walk(stmt)),
                       "registry_version": REGISTRY_VERSION})
    return {"factors": result, "excluded": excluded}


def factors(source, context_binding):
    return registry(source, context_binding)["factors"]


def _actual_ancestor_binding(tree, definition):
    """Reject displaced/possibly rebound definitions without gating unrelated code.

    This is intentionally bounded: a namespace operation cannot certify the
    historical slot. Ordinary conditionals concerning other names are allowed.
    """
    name = definition.name
    for node in ast.walk(tree):
        if isinstance(node, ast.Global) and name in node.names:
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC:
            return False
    for statement in tree.body:
        if statement is definition:
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if statement.name == name:
                return False
            continue  # Local variables in other function bodies do not bind this slot.
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return False
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bound = {a.asname or (a.name.split(".")[0] if isinstance(node, ast.Import) else a.name)
                         for a in node.names}
                if name in bound or "*" in bound:
                    return False
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
                return False
            if (isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                    and isinstance(node.value, ast.Name) and node.value.id == name):
                return False
    return True


def replace(source, factor, replacement, op="LOCAL_REWRITE"):
    """Splice exactly the received function bytes; reject rather than repair."""
    if op not in {"LOCAL_REWRITE", "LOCAL_PARAM"}:
        raise ContractError("NOT_A_LOCAL_OPERATOR")
    if sha(source) != factor["program_hash"]:
        raise ContractError("STALE_PARENT_SOURCE")
    current = next((f for f in factors(source, factor["context_binding"])
                    if f["start"] == factor["start"] and f["version"] == factor["version"]), None)
    if current is None or current["background"] != factor["background"]:
        raise ContractError("STALE_FACTOR_OR_BACKGROUND")
    if any(current[key] != factor[key] for key in ("name", "end", "source", "signature", "binding_manifest")):
        raise ContractError("STALE_OR_MUTATED_TARGET_DESCRIPTOR")
    tree = ast.parse(replacement)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ContractError("RESPONSE_MUST_BE_ONE_COMPLETE_FUNCTION")
    node = tree.body[0]
    if node.decorator_list or _signature(node) != factor["signature"]:
        raise ContractError("FUNCTION_SIGNATURE_CHANGED")
    if op == "LOCAL_PARAM":
        original = ast.parse(factor["source"]).body[0]
        if _body_param_shape(original) != _body_param_shape(node):
            raise ContractError("LOCAL_PARAM_CHANGED_NONNUMERIC_AST")
    data = source.encode("utf-8")
    child = (data[:factor["start"]] + replacement.encode("utf-8") + data[factor["end"]:]).decode("utf-8")
    child_factors = factors(child, factor["context_binding"])
    target = next((f for f in child_factors if f["name"] == factor["name"]), None)
    if target is None or target["signature"] != factor["signature"]:
        raise ContractError("REPLACEMENT_BINDINGS_NOT_IN_LOCAL_DOMAIN")
    # Existing referenced names retain their actual current module bindings.
    for name in set(factor["binding_manifest"]) & set(target["binding_manifest"]):
        old, new = factor["binding_manifest"][name], target["binding_manifest"][name]
        # Lines can move below an inserted body, but binding bytes cannot.
        if {k: v for k, v in old.items() if k != "line"} != {k: v for k, v in new.items() if k != "line"}:
            raise ContractError("EXISTING_REFERENCE_BINDING_CHANGED")
    return child


def compatible_reference(current_factor, ancestor_sources, current_source=None):
    """Nearest eligible *actually evaluated* ancestor, never quality selected.

    Ancestors are supplied nearest-first, or all have ``depth`` (then ordered
    by depth and source hash). Each is a record with source and explicit
    ``quality_available=True``. The caller alone owns lineage membership.
    """
    ancestors = list(ancestor_sources)
    if ancestors and all(isinstance(a, dict) and "depth" in a for a in ancestors):
        ancestors.sort(key=lambda a: (a["depth"], sha(a["source"])))
    for ancestor in ancestors:
        if not isinstance(ancestor, dict) or ancestor.get("quality_available") is not True:
            continue
        source = ancestor["source"]
        # Eligibility belongs to the current insertion context, not the old
        # module. An old module may contain unrelated dynamic code while this
        # actual helper is a valid reference in today's fixed module.
        tree = ast.parse(source)
        data = source.encode("utf-8")
        lines = data.splitlines(keepends=True)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        for node in definitions:
            if (node.name != current_factor["name"] or node.decorator_list or
                    getattr(node, "type_params", []) or
                    sum(n.name == node.name for n in definitions) != 1 or
                    not _actual_ancestor_binding(tree, node)):
                continue
            start, end = _span(node, lines)
            previous = {"name": node.name, "signature": _signature(node),
                        "source": data[start:end].decode("utf-8"), "start": start, "end": end,
                        "program_hash": sha(source), "reference_only": True}
            if (previous["name"] != current_factor["name"] or
                    previous["signature"] != current_factor["signature"] or
                    previous["source"] == current_factor["source"]):
                continue
            result = {"factor": previous, "source": source, "program_hash": sha(source),
                      "reference_source": previous["source"], "depth": ancestor.get("depth"),
                      "quality_available": True, "requires_current_complete_evaluation": True,
                      "requires_current_contract_check": current_source is None}
            if current_source is not None:
                try:
                    result["replacement_source"] = replace(current_source, current_factor, previous["source"])
                except (ContractError, SyntaxError, ValueError):
                    continue
            return result
    return None


def natural_pair(parent_source, child_source, context_binding):
    """Find an exact one-function undo. This certifies bytes, not evaluations."""
    if parent_source == child_source:
        return None
    parents, children = factors(parent_source, context_binding), factors(child_source, context_binding)
    for child in children:
        for parent in parents:
            if (parent["name"] != child["name"] or parent["signature"] != child["signature"]
                    or parent["source"] == child["source"]):
                continue
            try:
                undone = replace(child_source, child, parent["source"])
            except (ContractError, SyntaxError, ValueError):
                continue
            if undone == parent_source:
                return {"parent_factor": parent, "child_factor": child, "exact_undo": True,
                        "reference_program_hash": sha(parent_source), "program_hash": sha(child_source),
                        "context_binding": copy.deepcopy(context_binding),
                        "requires_matched_complete_evaluations": True}
    return None
