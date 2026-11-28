"""
Crash-safe JSON persistence for PulseAI.

Why this exists
---------------
The original code wrote JSON with ``open(path, "w")`` and, on *any* read error,
returned an empty list/dict.  A crash mid-write (or one transient read error)
could therefore silently wipe the whole trade ledger on the next save.

This module provides:
  * atomic_write_json  - write to a temp file, fsync, then os.replace().
                         A crash leaves either the old or the new file, never half a file.
                         The last known-good file is kept as ``<path>.bak``.
  * safe_load_json     - on a corrupt file: quarantine it, try the ``.bak``,
                         and for *critical* files raise StateCorruptionError
                         (fail closed) instead of starting from an empty state.
"""
import json
import os
import shutil
import tempfile
import time


class StateCorruptionError(RuntimeError):
    """Raised when a critical state file is unreadable and no valid backup exists."""


def _json_default(o):
    try:
        import numpy as np  # local import keeps this module dependency-light

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _type_ok(obj, expected_type):
    return expected_type is None or isinstance(obj, expected_type)


def atomic_write_json(path, obj, indent=2, keep_backup=True):
    """Atomically write ``obj`` as JSON to ``path``."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    fd, tmp_path = tempfile.mkstemp(prefix=base + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=_json_default)
            f.flush()
            os.fsync(f.fileno())

        # Keep the previous file as a backup, but only if it is itself valid JSON,
        # so a corrupt file can never overwrite a good backup.
        if keep_backup and os.path.exists(path):
            try:
                _read_json(path)
                shutil.copyfile(path, path + ".bak")
            except Exception:
                pass

        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _quarantine(path):
    try:
        dest = f"{path}.corrupt-{int(time.time())}"
        shutil.copyfile(path, dest)
        return dest
    except OSError:
        return None


def safe_load_json(path, default, expected_type=None, critical=False):
    """
    Load JSON from ``path``.

    * Missing file            -> ``default`` (a genuinely fresh start).
    * Corrupt/unreadable file -> quarantine a copy, try ``<path>.bak``;
                                 if that fails too: raise StateCorruptionError when
                                 ``critical`` else return ``default``.
    """
    if not os.path.exists(path):
        return default

    try:
        obj = _read_json(path)
        if _type_ok(obj, expected_type):
            return obj
        raise ValueError(f"unexpected type {type(obj).__name__}")
    except Exception as primary_err:
        quarantined = _quarantine(path)
        bak = path + ".bak"
        if os.path.exists(bak):
            try:
                obj = _read_json(bak)
                if _type_ok(obj, expected_type):
                    print(f"[STATE WARN] {path} unreadable ({primary_err}); restored from backup.")
                    return obj
            except Exception:
                pass
        if critical:
            raise StateCorruptionError(
                f"{path} is unreadable ({primary_err}) and no valid backup exists. "
                f"Quarantined copy: {quarantined}. Refusing to continue with empty state."
            ) from primary_err
        print(f"[STATE WARN] {path} unreadable ({primary_err}); using default.")
        return default