# DECADE: Credit-Guided Efficient Automatic Heuristic Design via Program Decomposition and Edit Attribution

DECADE studies automatic heuristic design through program decomposition and edit attribution. It represents a heuristic as editable program regions, records the measured effects of valid local edits, and uses compatible edit evidence to inform subsequent proposals. The included implementation targets constructive heuristics for the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP).

## Contents

- [Overview](#overview)
- [Method](#method)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Task interfaces](#task-interfaces)
- [Experiment execution](#experiment-execution)
- [Outputs](#outputs)

## Overview

Automatic heuristic design involves deciding both **what to change** in a program and **which previous changes provide useful evidence**. DECADE exposes several edit scopes and keeps each recorded effect tied to its actual program version and evaluation context.

The implementation provides:

- **Program decomposition:** an abstract syntax tree (AST) catalog of statement blocks, function groups, and whole-program edits.
- **Edit attribution:** exact before/after source tracking and paired feedback from the same evaluation instances.
- **Context-aware edit memory:** retrieval of compatible edit cases, including their scope, program background, and observed effect.
- **Budgeted search:** explicit candidate evaluation accounting, bounded technical repair, and saved search states.
- **Separated generation and evaluation:** model responses are parsed and assembled before a controlled evaluator measures complete programs.

The integrated driver in [`main_runtime.py`](main_runtime.py) runs the `F1` search path. The `fuse_*` package names and protocol identifiers are retained implementation names used by this snapshot.

## Method

### 1. Decompose the program

[`fuse_ane/regions.py`](fuse_ane/regions.py) builds the editable region catalog. The search uses three scopes:

| Scope | Editable unit | Purpose |
| --- | --- | --- |
| `B` | Statement block | Refine a bounded part of a function. |
| `C` | Function or related function group | Revise a larger unit while retaining the surrounding program. |
| `G` | Complete program | Explore a broader alternative or combine ideas from a donor program. |

The `F1` proposal schedule cycles through `B, B, C, C, ALT, MIX`. `ALT` and `MIX` use whole-program proposals; unavailable local scopes fall back to a larger scope. The host supplies a bounded menu, and the model chooses a permitted target within that menu.

### 2. Generate and evaluate edits

[`fuse_ane/sibling_view.py`](fuse_ane/sibling_view.py) prepares the current program, task contract, edit menu, and compact evidence cards. The [proposal templates](authority/shared_edit_siblings/prompts/) specify the response format. [`fuse_ane/sibling_protocol.py`](fuse_ane/sibling_protocol.py) parses and assembles the response before evaluation.

The integrated driver allows one normal proposal per step and at most one eligible technical repair per pair of normal proposal positions. Repair uses the frozen proposal context and the technical failure information.

### 3. Attribute measured effects

[`fuse_ane/edit_evidence.py`](fuse_ane/edit_evidence.py) checks that parent and child programs have complete, matched feedback on 64 instances under the same evaluation contract. A local attribution record additionally requires a valid replacement in one registered `B` or `C` region and unchanged source outside that region.

Whole-program changes, duplicates, and failed evaluations receive distinct record types. Retrieved historical effects remain associated with their original program context; they are evidence for proposing an edit, not measurements of the new candidate.

### 4. Continue within the evaluation budget

[`eval_budget.py`](eval_budget.py) counts fresh feedback evaluation attempts, including failures, while excluding exact cache hits. The integrated `F1` driver sets a budget of 200 attempts, including initialization, and records intermediate frontiers at 50, 100, 150, and 200 attempts. These values describe the supplied implementation configuration.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── run_study.py                    # Configuration checks and study launcher
├── configs/f1.example.json         # Example F1 configuration
├── main_runtime.py                 # Integrated F1 search adapter
├── common.py                       # Shared task contracts, evaluator, and client
├── eval_budget.py                  # Physical evaluation budget accounting
├── authority/shared_edit_siblings/
│   ├── prompts/                    # Proposal and technical-repair templates
│   └── reference/                  # Compact edit-contrast serialization
├── fuse_ane/                       # Decomposition, edit memory, and search policies
├── fuse_addon/                     # Transport, task data, and evaluator utilities
├── fuse_credit/                    # Credit, archive, and native initialization support
├── fuse_paired/                    # Paired evaluation and supporting adapters
├── fuse_patch/                     # Assembly, task contracts, and isolated execution
├── fuse_ig/                        # Credential-file validation
└── vendor/upstream_hifo/            # Included HiFo source and its license
```

## Getting started

Use Python 3.12 on Linux or macOS for the offline examples. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements cover the included Python modules and offline examples. Full experiments use Linux and Docker, with a compatible Python environment mounted into the candidate container.

### Inspect program decomposition

This example builds an edit catalog from a small TSP heuristic without executing a generated candidate or calling a model service.

```bash
python - <<'PY'
from fuse_addon.evaluation import digest
from fuse_ane.regions import catalog

source = """import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    distances = distance_matrix[current_node, unvisited_nodes]
    index = int(np.argmin(distances))
    return int(unvisited_nodes[index])
"""

regions = catalog(
    source, task="tsp", group="demo", arm="F1",
    episode=1, parent_sha=digest(source), bounded=False,
)
scopes = sorted({region["scope"] for region in regions})
assert scopes == ["B", "C", "G"]
print("Available edit scopes:", scopes)
PY
```

Expected output:

```text
Available edit scopes: ['B', 'C', 'G']
```

### Check exact replacement

The replacement primitive verifies the original bytes and keeps the rest of the source unchanged:

```bash
python - <<'PY'
from fuse_ane.edit_reference import Replacement, apply_exact

parent = b"def score(x):\n    return x + 1\n"
before = b"x + 1"
start = parent.index(before)
edit = Replacement(start, start + len(before), before, b"x + 2")
child = apply_exact(parent, [edit])
assert child == b"def score(x):\n    return x + 2\n"
print(child.decode(), end="")
PY
```

Expected output:

```python
def score(x):
    return x + 2
```

These examples exercise source-processing components. They do not run heuristic search or reproduce experimental scores.

## Task interfaces

Both tasks use a constructive next-node interface defined in [`common.py`](common.py).

### Traveling Salesman Problem

```python
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    ...
```

`current_node` and `destination_node` are integer node IDs; the fixed destination is node `0`. `unvisited_nodes` is a NumPy array in the evaluator's distance-neighbor order, and `distance_matrix` is the complete NumPy distance matrix. Return one permitted integer node from `unvisited_nodes`, excluding Boolean values.

### Capacitated Vehicle Routing Problem

```python
def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    ...
```

`unvisited_nodes` contains all unvisited customers, including those whose demand exceeds `rest_capacity`. Return a feasible unvisited customer, or the depot when the current node is not already the depot. The contract disallows Boolean returns and split deliveries.

The supplied task descriptions use Euclidean TSP with 50 nodes and CVRP with 50 customers, demands from 1 to 9, and vehicle capacity 40. [`fuse_ane/data.py`](fuse_ane/data.py) contains deterministic synthetic instance generation helpers.

## Experiment execution

[`run_study.py`](run_study.py) launches the `F1` path through `main_runtime.search(spec, cpu)`. Use a fresh checkout for each study; the launcher refuses to overwrite an existing study.

### Configure the environment

Copy the example configuration to a local file:

```bash
cp configs/f1.example.json configs/local.json
```

Edit `configs/local.json` for your environment:

| Field | Description |
| --- | --- |
| `run_id` | Unique name for the run directory. |
| `task` | `tsp` or `cvrp`. |
| `group` | Dataset group under `panels/<task>/<group>/`. |
| `search_seed` | Integer seed for search initialization. |
| `cpu` | One CPU available to the controller and Docker. |
| `runtime` | Absolute path to a Python environment compatible with the container image, including NumPy. |
| `image` | Local Docker image name or ID; the launcher records its resolved image ID. |
| `credential_file` | Path to a private service environment file. Relative paths are resolved from the repository root. |
| `api_base_url` | HTTPS model-service base URL, including its API prefix. |

The template uses neutral paths and an intentionally non-operational `api.example.invalid` endpoint. Prepare a Linux-compatible container image and Python environment, then replace these placeholders with your local settings. The evaluator mounts the Python environment at the same absolute path inside the container and runs candidates with network access disabled.

Create the credential file locally with the fields below and restrict its permissions to `0600`. Its `BASE_URL` must match `api_base_url` in the configuration.

```dotenv
BASE_URL=https://api.example.invalid/v1
API_KEY=replace-with-your-service-key
MODEL_NAME=DeepSeek-V4-Flash
```

```bash
chmod 600 .secrets/service.env
```

The source retains its model and response-protocol checks. A different model requires a corresponding configuration review.

### Check and launch

Check prerequisites without sending model requests or executing candidates:

```bash
python run_study.py --config configs/local.json --check
```

After the data release, place the feedback panels under `panels/<task>/<group>/feedback-*.json`. The launcher checks for 64 valid, uniquely identified feedback instances, the credential file, the selected CPU, and the local Docker image.

Start the configured study once these checks pass:

```bash
python run_study.py --config configs/local.json --run
```

`--run` sends model requests and evaluates candidates. The launcher records the runtime configuration and source/panel hashes under `engineering/`, then starts the existing 200-attempt search. The preflight checks configuration and prerequisites; it does not establish container compatibility or model-service availability through a trial run.

The repository uses an explicit `.gitignore` allowlist. Local configurations, credentials, generated data, runtime records, and virtual environments are excluded from ordinary Git additions.

## Outputs

The integrated driver writes records under `runs/<run-id>/`:

| Path | Contents |
| --- | --- |
| `programs/` | Candidate source files indexed by content hash. |
| `candidates/` | Candidate bindings, evaluation outcomes, and paired quality records. |
| `evaluations/` | Per-instance results and complete-program summaries. |
| `evaluation_budget/` | Evaluation attempt intents and accounting records. |
| `transport/` and `requests/` | Request payloads, responses, and usage accounting. |
| `batches/` | Frozen proposal contexts, repair records, and edit-memory updates. |
| `states/` and `checkpoints/` | Saved archive states and evaluation frontiers. |
| `END.json` | Termination status and final budget/cost summary. |
