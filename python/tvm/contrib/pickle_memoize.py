# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Memoize result of function via pickle, used for cache testcases."""

# pylint: disable=broad-except,superfluous-parens
import atexit
import os
import pathlib
import sys

import functools

try:
    import cPickle as pickle
except ImportError:
    import pickle


def _get_global_cache_dir() -> pathlib.Path:
    if "XDG_CACHE_HOME" in os.environ:
        cache_home = pathlib.Path(os.environ.get("XDG_CACHE_HOME"))
    else:
        cache_home = pathlib.Path.home().joinpath(".cache")
    return cache_home.joinpath("tvm", f"pkl_memoize_py{sys.version_info[0]}")


GLOBAL_CACHE_DIR = _get_global_cache_dir()


class Cache(object):
    """A cache object for result cache.

    Parameters
    ----------
    key: str
       The file key to the function
    save_at_exit: bool
        Whether save the cache to file when the program exits
    """

    cache_by_key = {}

    def __init__(self, key, save_at_exit):
        self._cache = None

        self.path = GLOBAL_CACHE_DIR.joinpath(key)
        self.dirty = False
        self.save_at_exit = save_at_exit

    @property
    def cache(self):
        """Return the cache, initializing on first use."""

        if self._cache is not None:
            return self._cache

        if self.path.exists():
            with self.path.open("rb") as cache_file:
                try:
                    cache = pickle.load(cache_file)
                except pickle.UnpicklingError:
                    cache = {}
        else:
            cache = {}

        self._cache = cache
        return self._cache

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self.path.open("wb") as out_file:
                pickle.dump(self.cache, out_file, pickle.HIGHEST_PROTOCOL)


@atexit.register
def _atexit():
    """Save handler."""
    for value in Cache.cache_by_key.values():
        if value.save_at_exit:
            value.save()


def memoize(key, save_at_exit=False):
    """Memoize the result of function and reuse multiple times.

    Parameters
    ----------
    key: str
        The unique key to the file
    save_at_exit: bool
        Whether save the cache to file when the program exits

    Returns
    -------
    fmemoize : function
        The decorator function to perform memoization.
    """

    def _register(f):
        """Registration function"""
        allow_types = (str, int, float, tuple)
        fkey = key + "." + f.__name__ + ".pkl"
        if fkey not in Cache.cache_by_key:
            Cache.cache_by_key[fkey] = Cache(fkey, save_at_exit)
        cache = Cache.cache_by_key[fkey]
        cargs = tuple(x.cell_contents for x in f.__closure__) if f.__closure__ else ()
        cargs = (len(cargs),) + cargs

        @functools.wraps(f)
        def _memoized_f(*args, **kwargs):
            assert not kwargs, "Only allow positional call"
            key = cargs + args
            for arg in key:
                if isinstance(arg, tuple):
                    for x in arg:
                        assert isinstance(x, allow_types)
                else:
                    assert isinstance(arg, allow_types)
            if key in cache.cache:
                return cache.cache[key]
            res = f(*args)
            cache.cache[key] = res
            cache.dirty = True
            return res

        return _memoized_f

    return _register
