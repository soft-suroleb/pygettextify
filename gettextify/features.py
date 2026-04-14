import re
import json
import string
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

FEATURE_NAMES = [
    "with_format", "with_inner_format", "count", "length",
    "special_letters", "is_empty", "last_punctuation", "is_capital",
    "spaces", "underscore", "is_upper", "is_lower",
    "snake_case", "camel_case", "json", "xml", "html",
    "path", "in_stoplist", "global_count",
]

STOPLIST = {
    'VERSION', 'DEBUG', 'TEMP', 'TEST', 'STUB', 'MOCK', 'SAMPLE', 'DUMMY',
    'INTERNAL', 'DEV', 'BUILD', 'SNAPSHOT', 'RELEASE', 'ALPHA', 'BETA',
    'RC', 'PATCH', 'DEFAULT', 'NULL', 'TRUE', 'FALSE', 'ON', 'OFF',
    'ENABLED', 'DISABLED', 'OK', 'ERROR', 'FAIL', 'PASSED', 'SUCCESS',
    'WARNING', 'INFO', 'LOG', 'EXCEPTION', 'USER_ID', 'API_KEY', 'SECRET',
    'TOKEN', 'UUID', 'SESSION', 'PORT', 'HOST', 'PATH', 'CONFIG', 'TIMEOUT',
    'RETRY', 'ATTEMPT', 'LIMIT', 'MIN', 'MAX', 'COUNT', 'LENGTH', 'SIZE',
    'INDEX', 'OFFSET', 'BUFFER', 'CACHE', 'FLAG', 'MODE', 'TYPE', 'ID',
    'UID', 'PID', 'SID', 'CID', 'EXT', 'DIR', 'FOLDER', 'FILE', 'EXTENSION',
    'URL', 'URI', 'HTTP', 'HTTPS', 'JSON', 'XML', 'SQL', 'CSV', 'TSV',
    'YAML', 'ENV', 'LOCAL', 'REMOTE', 'ROOT', 'HOME', 'ADMIN', 'SYSTEM',
    'MACHINE', 'AUTO', 'MANUAL', 'STATIC', 'DYNAMIC', 'RANDOM', 'SEED',
    'INIT', 'SETUP', 'SHUTDOWN', 'CLEANUP', 'RESET', 'START', 'STOP',
    'BEGIN', 'END', 'OPEN', 'CLOSE', 'READ', 'WRITE', 'UPDATE', 'DELETE',
    'INSERT', 'SAVE', 'LOAD', 'IMPORT', 'EXPORT', 'SYNC', 'CONNECT',
    'DISCONNECT',
}

_PUNCTUATION = set(string.punctuation)


def _is_snake_case(key: str) -> bool:
    return (
        bool(re.fullmatch(r'[a-z]+(_[a-z0-9]+)*', key))
        or bool(re.fullmatch(r'[A-Z]+(_[A-Z0-9]+)*', key))
    )


def _is_camel_case(key: str) -> bool:
    return bool(re.fullmatch(r'[a-zA-Z]+(?:[A-Z][a-z0-9]*)*', key))


def _is_json(key: str) -> bool:
    try:
        json.loads(key)
        return True
    except Exception:
        return False


def _is_xml(key: str) -> bool:
    try:
        ET.fromstring(key)
        return True
    except Exception:
        return False


def _is_html(key: str) -> bool:
    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.found = False

        def handle_starttag(self, tag, attrs):
            self.found = True

    p = _Parser()
    try:
        p.feed(key)
        return p.found
    except Exception:
        return False


_WIN_PATH_RE = re.compile(r'^[A-Za-z]:\\')


def _looks_like_path(key: str) -> bool:
    if _WIN_PATH_RE.match(key):
        return True
    try:
        p = Path(key)
        return len(p.parts) > 1 or p.suffix != ""
    except Exception:
        return False


_INNER_FORMAT_RE = re.compile(
    r'%\([a-zA-Z_]\w*\)[sdifoxXeEgGcr]'
    r'|%[YmdHMSfzZaAbBpIjUWwxXcG]'
    r'|\{[a-zA-Z_]\w*(?:![rsaRS])?(?::[^}]*)?\}'
    r'|\{\d+(?::[^}]*)?\}'
)


def _has_inner_format(key: str) -> bool:
    return bool(_INNER_FORMAT_RE.search(key))


def compute_features(
    key: str,
    with_format: bool,
    count: int,
    global_count: int = 1,
) -> dict:
    length = len(key)
    special_letters = sum(1 for c in key if c in _PUNCTUATION)
    spaces = key.count(' ')
    underscore = key.count('_')
    is_empty = len(key.strip()) == 0
    last_punct = (not is_empty) and key.strip()[-1] in '.!?'
    is_capital = length > 1 and key[0].isupper() and key[1].islower()
    wf = int(with_format)

    return {
        "with_format": wf,
        "with_inner_format": int(_has_inner_format(key)),
        "count": count,
        "length": length,
        "special_letters": special_letters,
        "is_empty": int(is_empty),
        "last_punctuation": int(last_punct),
        "is_capital": int(is_capital),
        "spaces": spaces,
        "underscore": underscore,
        "is_upper": int(key.isupper()),
        "is_lower": int(key.islower()),
        "snake_case": int(_is_snake_case(key)),
        "camel_case": int(_is_camel_case(key)),
        "json": int(_is_json(key)),
        "xml": int(_is_xml(key)),
        "html": int(_is_html(key)),
        "path": int(_looks_like_path(key)),
        "in_stoplist": int(key.upper() in STOPLIST),
        "global_count": global_count,
    }
