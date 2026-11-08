"""Untrusted per-instance V3.1 candidate wrapper, mounted at /rpc/worker.py.

The entire process, including this module and PROTOCOL_FD, is candidate-owned.
Only the external host validates nodes, constructs/scores routes, bounds output,
enforces deadlines, and writes scientific facts. This is not a security oracle.
Importing the module is inert; main() is used only inside the remote container.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
import types


CANDIDATE_SOURCE = '/candidate/source.py'
PROTOCOL_FD = None
TASKS = ('TSP-Construct', 'CVRP-Construct')


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('DUPLICATE_HOST_JSON_KEY:' + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('NONFINITE_HOST_JSON_NUMBER:' + value)


def parse_command(line):
    """Decode one host-owned JSON line without coercion or repair."""
    if not isinstance(line, bytes) or not line.endswith(b'\n'):
        raise ValueError('HOST_COMMAND_MUST_BE_ONE_UTF8_JSON_LINE')
    value = json.loads(line.decode('utf-8', errors='strict'),
                       object_pairs_hook=_json_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError('HOST_COMMAND_MUST_BE_OBJECT')
    return value


def validate_init(command):
    task = command.get('task')
    if task not in TASKS:
        raise ValueError('UNKNOWN_TASK')
    keys = {'task', 'execution_seed', 'distance_matrix'}
    if task == 'CVRP-Construct':
        keys.add('demands')
    if set(command) != keys:
        raise ValueError('INIT_EXACT_KEYS_REQUIRED')
    seed = command['execution_seed']
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('INVALID_INSTANCE_EXECUTION_SEED')
    matrix = command['distance_matrix']
    if (not isinstance(matrix, list) or len(matrix) < 2 or
            any(not isinstance(row, list) or len(row) != len(matrix) or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in row) for row in matrix)):
        raise ValueError('INVALID_CURRENT_INSTANCE_DISTANCE_MATRIX')
    if task == 'CVRP-Construct':
        demands = command['demands']
        if (not isinstance(demands, list) or len(demands) != len(matrix) or
                any(type(value) is not int or not -(2**63) <= value < 2**63
                    for value in demands)):
            raise ValueError('INVALID_CURRENT_INSTANCE_DEMANDS')
    return command


def instance_arrays(command, np):
    """Private current-instance numeric arrays; no panel or host memory view."""
    validate_init(command)
    return {'task': command['task'], 'execution_seed': command['execution_seed'],
            'distance_matrix': np.asarray(command['distance_matrix'], dtype=np.float64).copy(),
            'demands': np.asarray(command['demands'], dtype=np.int64).copy()
                       if command['task'] == 'CVRP-Construct' else None}


def validate_step(command, task):
    keys = {'seq', 'current', 'destination', 'unvisited'}
    if task == 'CVRP-Construct':
        keys.add('remaining_capacity')
    elif task != 'TSP-Construct':
        raise ValueError('UNKNOWN_TASK')
    if set(command) != keys:
        raise ValueError('STEP_EXACT_KEYS_REQUIRED')
    if (any(type(command[key]) is not int for key in ('seq', 'current', 'destination')) or
            command['seq'] < 0 or command['destination'] != 0):
        raise ValueError('INVALID_STEP_INTEGER_FIELDS')
    if (not isinstance(command['unvisited'], list) or
            any(type(node) is not int for node in command['unvisited'])):
        raise ValueError('INVALID_STEP_UNVISITED_ARRAY')
    if task == 'CVRP-Construct' and type(command['remaining_capacity']) is not int:
        raise ValueError('INVALID_STEP_CAPACITY')
    return command


def call_arguments(command, state, np):
    """Preserve original positional ABI, dtypes, order and ordinary copy rules."""
    validate_step(command, state['task'])
    unvisited = np.asarray(command['unvisited'], dtype=int)
    if state['task'] == 'TSP-Construct':
        return (command['current'], command['destination'], unvisited,
                state['distance_matrix'])
    # The prior CVRP host supplied fresh read-only array copies on each call.
    # These remain ordinary private Python/NumPy objects, not a security wall.
    demands, matrix = state['demands'].copy(), state['distance_matrix'].copy()
    unvisited.setflags(write=False)
    demands.setflags(write=False)
    matrix.setflags(write=False)
    return (command['current'], command['destination'], unvisited,
            command['remaining_capacity'], demands, matrix)


def serializable_node(node, np):
    """NumPy ABI scalars only; never turn bool/float into a legal integer."""
    if isinstance(node, np.bool_):
        return bool(node)
    if isinstance(node, np.integer):
        return int(node)
    if isinstance(node, np.floating):
        return float(node)
    return node


def write_frame(fd, payload):
    """Untrusted wire writer. The external host enforces its own frame bound."""
    data = (json.dumps(payload, ensure_ascii=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')
    cursor = 0
    while cursor < len(data):
        sent = os.write(fd, data[cursor:])
        if sent <= 0:
            raise BrokenPipeError('RPC_PROTOCOL_WRITE_FAILED')
        cursor += sent


def load_candidate(np):
    """Read the one immutable mounted source without newline normalization."""
    source_bytes = Path(CANDIDATE_SOURCE).read_bytes()
    source = source_bytes.decode('utf-8', errors='strict')
    module = types.ModuleType('native_candidate')
    module.__file__ = CANDIDATE_SOURCE
    module.__dict__['np'] = np
    sys.modules[module.__name__] = module
    # Full Python is intentional; the host is in a different trust boundary.
    exec(compile(source, CANDIDATE_SOURCE, 'exec'), module.__dict__, module.__dict__)
    function = module.__dict__.get('select_next_node')
    if not callable(function):
        raise ValueError('MISSING_SELECT_NEXT_NODE_ABI')
    return function


def serve(protocol_fd):
    first = sys.stdin.buffer.readline()
    if not first:
        raise ValueError('MISSING_INSTANCE_INIT')
    command = validate_init(parse_command(first))
    import numpy as np
    state = instance_arrays(command, np)
    random.seed(state['execution_seed'])
    np.random.seed(state['execution_seed'])
    function = load_candidate(np)
    # Only one ordinary ready frame; it is not trusted source-loading timing.
    write_frame(protocol_fd, {'ready': True})
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return
        command = parse_command(line)
        arguments = call_arguments(command, state, np)
        node = function(*arguments)
        write_frame(protocol_fd, {'seq': command['seq'], 'node': serializable_node(node, np)})


def main():
    global PROTOCOL_FD
    # Ordinary Python and child-process stdout/stderr are diagnostics. A
    # malicious candidate can still find/alter this wrapper's protocol fd;
    # the host therefore treats EVERY returned byte as untrusted.
    PROTOCOL_FD = os.dup(1)
    os.dup2(2, 1)
    try:
        serve(PROTOCOL_FD)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        os.close(PROTOCOL_FD)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
