# Filename globbing utility
# Modified from https://github.com/python/cpython/blob/main/Lib/glob.py
# under license: https://github.com/python/cpython/blob/main/LICENSE
# Extended features:
#   1. Natural sorting of paths using the natsort package
#   2. Integration with mmengine's BaseStorageBackend to support various storage backends

from typing import Callable, TypeVar
import contextlib
import os
import re
import fnmatch
import sys
from typing import Optional
from mmengine.fileio import get_file_backend

from natsort import natsorted

from natsort.ns_enum import NSType, ns
from natsort.utils import NatsortInType

__all__ = ["glob", "iglob", "escape", "natsort_iglob"]

T = TypeVar("T")


def natsort_iglob(pathname, *, recursive=False, backend_args: Optional[dict] = None,
                 key: Optional[Callable[[T], NatsortInType]] = None,
                 reverse: bool = False,
                 alg: NSType = ns.DEFAULT):
    glob_data = iglob(pathname, recursive=recursive, backend_args=backend_args)
    return natsorted(glob_data, key, reverse, alg)


def glob(pathname, *, recursive=False, backend_args: Optional[dict] = None):
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    return list(iglob(pathname, recursive=recursive, backend_args=backend_args))


def iglob(pathname, *, recursive=False, backend_args: Optional[dict] = None):
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    sys.audit("glob.glob", pathname, recursive)
    backend = get_file_backend(pathname, backend_args=backend_args)
    it = _iglob(pathname, recursive, False, backend)
    if recursive and _isrecursive(pathname):
        s = next(it)  # skip empty string
        assert not s
    return it


def _iglob(pathname, recursive, dironly, backend):
    dirname, basename = os.path.split(pathname)
    if not has_magic(pathname):
        assert not dironly
        if basename:
            if backend.exists(pathname):
                yield pathname
        else:
            if backend.isdir(dirname):
                yield pathname
        return
    if not dirname:
        if recursive and _isrecursive(basename):
            yield from _glob2(dirname, basename, dironly, backend)
        else:
            yield from _glob1(dirname, basename, dironly, backend)
        return
    if dirname != pathname and has_magic(dirname):
        dirs = _iglob(dirname, recursive, True, backend)
    else:
        dirs = [dirname]
    if has_magic(basename):
        if recursive and _isrecursive(basename):
            glob_in_dir = _glob2
        else:
            glob_in_dir = _glob1
    else:
        glob_in_dir = _glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename, dironly, backend):
            yield os.path.join(dirname, name)


def _glob1(dirname, pattern, dironly, backend):
    names = _listdir(dirname, dironly, backend)
    if not _ishidden(pattern):
        names = (x for x in names if not _ishidden(x))
    return fnmatch.filter(names, pattern)


def _glob0(dirname, basename, dironly, backend):
    if not basename:
        if backend.isdir(dirname):
            return [basename]
    else:
        if backend.exists(os.path.join(dirname, basename)):
            return [basename]
    return []


def glob0(dirname, pattern, backend):
    return _glob0(dirname, pattern, False, backend)


def glob1(dirname, pattern, backend):
    return _glob1(dirname, pattern, False, backend)


def _glob2(dirname, pattern, dironly, backend):
    assert _isrecursive(pattern)
    yield pattern[:0]
    yield from _rlistdir(dirname, dironly, backend)


def _iterdir(dirname, dironly, backend):
    if not dirname:
        dirname = os.curdir
    try:
        entries = backend.list_dir_or_file(dirname, list_dir=True, list_file=not dironly)
        for entry in entries:
            name = os.path.basename(entry)
            if not dironly or backend.isdir(os.path.join(dirname, name)):
                yield name
    except OSError:
        return


def _listdir(dirname, dironly, backend):
    with contextlib.closing(_iterdir(dirname, dironly, backend)) as it:
        return list(it)


def _rlistdir(dirname, dironly, backend):
    names = _listdir(dirname, dironly, backend)
    for x in names:
        if not _ishidden(x):
            yield x
            path = os.path.join(dirname, x) if dirname else x
            for y in _rlistdir(path, dironly, backend):
                yield os.path.join(x, y)


magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')


def has_magic(s):
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None


def _ishidden(path):
    return path[0] in ('.', b'.'[0])


def _isrecursive(pattern):
    if isinstance(pattern, bytes):
        return pattern == b'**'
    else:
        return pattern == '**'


def escape(pathname):
    """Escape all special characters.
    """
    drive, pathname = os.path.splitdrive(pathname)
    if isinstance(pathname, bytes):
        pathname = magic_check_bytes.sub(br'[\1]', pathname)
    else:
        pathname = magic_check.sub(r'[\1]', pathname)
    return drive + pathname
