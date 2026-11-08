"""Three reads of existing core bytes; never recovery by execution or RAW walks."""
import json
import os
import time
from pathlib import Path

STUDY = None


def bytes_read(path):
    for attempt in range(1, 4):
        try:
            return Path(path).read_bytes()
        except OSError as exc:
            if STUDY is not None and STUDY.exists():
                folder = STUDY/'engineering'
                folder.mkdir(exist_ok=True)
                line = json.dumps(dict(path=str(path), attempt=attempt, errno=exc.errno,
                                       error=str(exc), unix=time.time()))+'\n'
                fd = os.open(folder/'READ_ERRORS.jsonl', os.O_CREAT|os.O_APPEND|os.O_WRONLY, 0o600)
                try:os.write(fd, line.encode())
                finally:os.close(fd)
            if attempt == 3:raise
            time.sleep(.2*attempt)


def read(path):
    return json.loads(bytes_read(path))


def install(study):
    global STUDY
    STUDY = Path(study)
    from fuse_addon import evaluation, transport
    evaluation.read = read
    transport.read = read
