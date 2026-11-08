"""Study-specific credential parsing for the user-authorized gateway change."""
from pathlib import Path
import os
import shlex
import stat
from urllib.parse import urlsplit

CREDENTIAL_FILE = Path(os.environ.get('DECADE_CREDENTIAL_FILE', '/opt/decade/config/service.env')).absolute()
BASE_URL = os.environ.get('DECADE_API_BASE_URL', 'https://api.example.invalid/v1')
MODEL = 'DeepSeek-V4-Flash'


def parse(text):
    url = urlsplit(BASE_URL)
    if (url.scheme != 'https' or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError('INVALID_SERVICE_BASE_URL')
    values = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, sep, raw = line.strip().removeprefix('export ').partition('=')
        if not sep or key.strip() not in ('BASE_URL', 'API_KEY', 'MODEL_NAME'):
            raise ValueError('UNEXPECTED_CREDENTIAL_FIELD')
        key = key.strip()
        if key in values:
            raise ValueError('DUPLICATE_CREDENTIAL_FIELD')
        parts = shlex.split(raw, comments=True)
        if len(parts) != 1 or not parts[0] or any(c.isspace() for c in parts[0]):
            raise ValueError('INVALID_CREDENTIAL_FIELD')
        value = parts[0]
        if any(token in value for token in ('${', '$(', '<', '>')):
            raise ValueError('UNEXPANDED_CREDENTIAL_PLACEHOLDER')
        values[key] = value
    if set(values) != {'BASE_URL', 'API_KEY', 'MODEL_NAME'}:
        raise ValueError('MISSING_CREDENTIAL_FIELD')
    if values['API_KEY'].lower().startswith('bearer'):
        raise ValueError('EXTRANEOUS_BEARER_PREFIX')
    if values['BASE_URL'] != BASE_URL or values['MODEL_NAME'] not in (MODEL, 'DeepSeek-V4.1-Flash'):
        raise ValueError('AUTHORIZED_RECOVERY_SERVICE_BINDING_CONFLICT')
    return values


def credentials(path):
    path = Path(path)
    if path != CREDENTIAL_FILE or path.is_symlink():
        raise ValueError('STUDY_CREDENTIAL_PATH_CONFLICT')
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError('STUDY_CREDENTIAL_MODE_CONFLICT')
    values = parse(path.read_text())
    url = urlsplit(values['BASE_URL'])
    return values, url.netloc, url.path.rstrip('/') + '/chat/completions'
