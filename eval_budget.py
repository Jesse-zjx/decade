"""A physical Feedback evaluation budget, independent of native algorithm counters."""
import hashlib
import json
import threading
import time
from pathlib import Path


class RunSafetyLimit(BaseException):
    pass


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)


class EvaluationBudget:
    def __init__(self, evaluator, client, run, limit, stop_exception):
        self.inner = evaluator
        self.client = client
        self.run = Path(run)
        self.limit = limit
        self.stop_exception = stop_exception
        self.started = 0
        self.completed = 0
        self.last_progress_calls = 0
        self.lock = threading.RLock()
        self.admission_lock = threading.RLock()
        self.source_indices = {}

    def guard_request(self):
        # Called under the client's request lock. Never take the evaluator lock
        # here: evaluation and an in-flight request may overlap under native EoH.
        if self.started >= self.limit:
            raise self.stop_exception()
        calls = len(self.client.requests)
        if calls >= 20 * self.limit:
            raise RunSafetyLimit('PHYSICAL_REQUEST_SAFETY_LIMIT')
        if calls - self.last_progress_calls >= 200:
            raise RunSafetyLimit('200_REQUESTS_WITHOUT_NEW_EVALUATION')

    def evaluate(self, source, split='feedback', owner=None):
        if split != 'feedback':
            return self.inner.evaluate(source, split, owner)
        with self.lock:
            h = hashlib.sha256(source.encode()).hexdigest()
            cache = self.run / 'evaluations' / split / h / 'SUMMARY.json'
            if cache.exists():
                result = self.inner.evaluate(source, split, owner)
                return dict(result, evaluation_index=self.source_indices[h],
                            counts_against_budget=False)
            with self.admission_lock:
                if self.started >= self.limit:
                    raise self.stop_exception()
                self.started += 1
                index = self.started
            self.source_indices[h] = index
            folder = self.run / 'evaluation_budget' / f'{index:06}'
            freeze(folder / 'INTENT.json', dict(evaluation_index=index, source_sha256=h,
                   owner=owner, started_unix=time.time(), budget=self.limit))
            try:
                result = self.inner.evaluate(source, split, owner)
            except BaseException as exc:
                freeze(folder / 'INTERRUPTED.json', dict(type=type(exc).__name__,
                       error=str(exc), ended_unix=time.time(), counts_against_budget=True))
                raise
            self.completed += 1
            self.last_progress_calls = len(self.client.requests)
            # Include every already-submitted call, including outstanding EoH
            # calls. Its eventual full bill is charged to this committed-cost
            # checkpoint; its future candidate is NOT made eligible here.
            committed = sorted(int(p.parent.name) for p in
                               (self.run / 'requests').glob('*/INTENT.json'))
            result = dict(result, evaluation_index=index, counts_against_budget=True)
            freeze(folder / 'RESULT.json', dict(result=result, completed_unix=time.time(),
                   committed_request_ids=committed, completed_evaluations=self.completed))
            return result

    def snapshot(self):
        return dict(limit=self.limit, started=self.started, completed=self.completed,
                    unit='fresh complete-Feedback evaluation attempt; failures count; cache hits do not')
