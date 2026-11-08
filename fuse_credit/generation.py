"""Frozen prompts and faithful response consumption; no requests or execution.

The macro sentences follow EoH v0.1 eoh_evolution.py (4b2338fc2317c743cec50c5c7a2e4684707af973).
The common module-format appendix and selected-function interface are the
explicit v1 execution adaptation. Policy labels and credit never enter prompts.
"""
from __future__ import annotations

import ast
import json

from fuse_addon.response_parser import extract_response
from .factors import ContractError, ENTRY_NAME, replace, sha

PROMPT_VERSION = "eoh-v01-complete-module-directed-v1"
TASK = (
    "Given a set of nodes with their coordinates, you need to find the shortest route "
    "that visits each node once and returns to the starting node. The task can be "
    "solved step-by-step by starting from the current node and iteratively choosing "
    "the next node. Help me design a novel algorithm that is different from the "
    "algorithms in literature to select the next node in each step."
)
SPEC = (
    "implement it in Python as a function named select_next_node. This function "
    "should accept 4 input(s): 'current_node', 'destination_node', 'univisited_nodes', "
    "'distance_matrix'. The function should return 1 output(s): 'next_node'. "
    "'current_node', 'destination_node', 'next_node', and 'unvisited_nodes' are node "
    "IDs. 'distance_matrix' is the distance matrix of nodes. All are Numpy arrays.\n"
    "Do not give additional explanations."
)
MODULE_FORMAT = (
    "\nExecution/output contract: current_node and destination_node are integer node IDs; "
    "the third argument is a NumPy array of currently allowed unvisited node IDs and "
    "distance_matrix is the NumPy pairwise distance array. Return an integer node ID "
    "from that provided array. Output exactly one complete Python module in one "
    "```python code block, including all imports and helper definitions it needs. "
    "All top-level functions execute in one module namespace. You may add, remove, "
    "merge or split helpers and change internal composition. Do not output fragments "
    "or multiple alternatives. No external I/O, files or network are available."
)
_FIRST = (
    "First, describe your new algorithm and main steps in one sentence. "
    "The description must be inside a brace. Next, "
)


def init_prompt():
    """Native i1 meaning, complete module format, nonbinding helper suggestion."""
    return (TASK + "\n" + _FIRST + SPEC + MODULE_FORMAT +
            "\nWhen useful, organize the implementation into 2–6 meaningful top-level "
            "helper functions. This is a suggestion, not a required count or a "
            "restriction on the algorithm's composition.")


def _parents(action, programs):
    result = []
    for identity in action["parents"]:
        program = programs[identity]
        if sha(program["source"]) != identity:
            raise ContractError("PARENT_HASH_NOT_ACTUAL_SOURCE")
        result.append(program)
    expected = 2 if action["op"] in {"MACRO_E1", "MACRO_E2"} else 1
    if len(result) != expected:
        raise ContractError("WRONG_NUMBER_OF_ACTUAL_PARENTS")
    return result


def render_prompt(action, programs):
    """Render only real parents and the selected action; ignore policy metadata."""
    op = action["op"]
    if op in {"i1", "I1"}:
        return init_prompt()
    parents = _parents(action, programs)
    if op in {"LOCAL_REWRITE", "LOCAL_PARAM"}:
        target = action["target"]
        source = parents[0]["source"]
        if target["program_hash"] != sha(source):
            raise ContractError("TARGET_NOT_BOUND_TO_ACTUAL_PARENT")
        # Only the immutable source identity and selected target are sent. The
        # full action dict may carry private decision scores; never serialize it.
        contract = {
            "operation": op, "name": target["name"], "signature": target["signature"],
            "target_start_utf8_byte": target["start"], "target_end_utf8_byte": target["end"],
            "immutable_ranges_utf8_bytes": [[0, target["start"]],
                                             [target["end"], len(source.encode("utf-8"))]],
        }
        instruction = (
            "Rewrite the selected complete helper function, preserving its signature. "
            "You may change its local algorithm logic."
            if op == "LOCAL_REWRITE" else
            "Change only finite int/float constant values inside the selected function body. "
            "Booleans are not numeric parameters. After ignoring those numeric values, "
            "the AST must be unchanged; do not alter rules, operators or control flow."
        )
        return (
            TASK + "\nActual complete parent module:\n```python\n" + source + "\n```\n" +
            "Selected action and immutable ranges:\n" + json.dumps(contract, sort_keys=True) +
            "\nOriginal selected complete function:\n```python\n" + target["source"] + "\n```\n" +
            instruction + " All non-target source bytes are immutable. Existing name bindings "
            "must be preserved; every reference must resolve in the current module. "
            "Output a one-sentence algorithm description in braces followed by exactly one "
            "```python code block containing only one complete definition of the selected "
            "function. Do not reproduce the module, add top-level definitions/imports, "
            "change the entry function, or give alternative solutions. The host will splice "
            "your exact function bytes into the selected range without repairing them."
        )
    if op in {"MACRO_E1", "MACRO_E2"}:
        block = "".join("No." + str(i + 1) + " algorithm and the corresponding code are: \n" +
                        p.get("algorithm", "") + "\n" + p["source"] + "\n"
                        for i, p in enumerate(parents))
        start = TASK + "\nI have 2 existing algorithms with their codes as follows: \n" + block
        if op == "MACRO_E1":
            body = ("Please help me create a new algorithm that has a totally different form "
                    "from the given ones. \n" + _FIRST)
        else:
            body = ("Please help me create a new algorithm that has a totally different form "
                    "from the given ones but can be motivated from them. \nFirstly, identify "
                    "the common backbone idea in the provided algorithms. Secondly, based "
                    "on the backbone idea describe your new algorithm in one sentence. "
                    "The description must be inside a brace. Thirdly, ")
    elif op in {"MACRO_M1", "MACRO_M2"}:
        parent = parents[0]
        start = (TASK + "\nI have one algorithm with its code as follows. Algorithm description: " +
                 parent.get("algorithm", "") + "\nCode:\n" + parent["source"] + "\n")
        body = (
            "Please assist me in creating a new algorithm that has a different form but can "
            "be a modified version of the algorithm provided. \n"
            if op == "MACRO_M1" else
            "Please identify the main algorithm parameters and assist me in creating a new "
            "algorithm that has a different parameter settings of the score function provided. \n"
        ) + _FIRST
    else:
        raise ContractError("UNKNOWN_OPERATOR:" + str(op))
    return start + body + SPEC + MODULE_FORMAT


def consume_response(response, action, programs):
    """Return parsed bytes and contract evidence, never retry or fix a response.

    ``source`` is the actual full successor iff ``valid``. ``response_source``
    preserves the extracted bytes even for a failed local or syntax contract.
    Quality and actual entry behavior remain the remote evaluator's task.
    """
    parsed = extract_response(response, ENTRY_NAME)
    result = {"valid": False, "status": parsed.status, "source": None,
              "response_source": parsed.source, "algorithm": parsed.description,
              "parse": parsed.evidence(), "prompt_version": PROMPT_VERSION,
              "error": None}
    if parsed.source is None:
        return result
    try:
        ast.parse(parsed.source)
        if action["op"] in {"LOCAL_REWRITE", "LOCAL_PARAM"}:
            parents = _parents(action, programs)
            source = replace(parents[0]["source"], action["target"], parsed.source, action["op"])
            target = action["target"]
            result["splice"] = {"start_utf8_byte": target["start"], "end_utf8_byte": target["end"],
                                "replacement_sha256": sha(parsed.source),
                                "non_target_bytes_preserved": True}
        elif action["op"] in {"i1", "I1", "MACRO_M1", "MACRO_M2", "MACRO_E1", "MACRO_E2"}:
            source = parsed.source
        else:
            raise ContractError("UNKNOWN_OPERATOR:" + str(action["op"]))
    except (ContractError, SyntaxError, ValueError, TypeError) as exc:
        result["status"] = "CONTRACT_FAILURE"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return result
    result.update(valid=True, status="CONTRACT_ACCEPTED", source=source, source_sha256=sha(source))
    return result
