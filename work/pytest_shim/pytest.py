"""Minimal pytest-compatible shim for running NetworkX community tests
in an environment without pytest.  Supports only the features used by
the community test suite: approx, raises, mark.parametrize/skip,
fixture, and plain assert-based test functions/classes.
"""

import importlib
import inspect
import math
import sys
import traceback
import unittest
from contextlib import contextmanager


class Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = 1e-6 if rel is None else rel
        self.abs = 1e-12 if abs is None else abs

    def __eq__(self, actual):
        try:
            if isinstance(self.expected, (list, tuple, set)):
                if not isinstance(actual, type(self.expected)) and not isinstance(
                    self.expected, type(actual)
                ):
                    if len(actual) != len(self.expected):
                        return False
                tol = max(self.abs, self.rel * max(map(abs, self.expected), default=0.0))
                return all(
                    any(math.isclose(a, b, rel_tol=0.0, abs_tol=tol) for b in self.expected)
                    for a in actual
                )
            tol = max(self.abs, self.rel * abs(self.expected))
            return math.isclose(actual, self.expected, rel_tol=0.0, abs_tol=tol)
        except TypeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"approx({self.expected!r})"


def approx(expected, rel=None, abs=None):
    return Approx(expected, rel, abs)


class _Raises:
    def __init__(self, expected_exception, match=None):
        self.expected = expected_exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(
                f"DID NOT RAISE {self.expected.__name__}"
            )
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None and self.match not in str(exc):
            raise AssertionError(
                f"pattern {self.match!r} not found in {str(exc)!r}"
            )
        return True


def raises(expected_exception, match=None, *args, **kwargs):
    if callable(match):
        # pytest.raises(exc, func, *args, **kwargs) call form
        func = match
        try:
            func(*args, **kwargs)
        except expected_exception as exc:
            return None
        raise AssertionError(f"DID NOT RAISE {expected_exception.__name__}")
    if args or kwargs:
        func = args[0]
        rest = args[1:]
        try:
            func(*rest, **kwargs)
        except expected_exception as exc:
            if match is not None and match not in str(exc):
                raise AssertionError(
                    f"pattern {match!r} not found in {str(exc)!r}"
                )
            return None
        raise AssertionError(f"DID NOT RAISE {expected_exception.__name__}")
    return _Raises(expected_exception, match)


def importorskip(module_name, reason=""):
    try:
        return __import__(module_name)
    except ImportError:
        import unittest

        raise unittest.SkipTest(f"optional dependency {module_name} missing")


class _Mark:
    def parametrize(self, argnames, argvalues):
        if isinstance(argnames, str):
            names = [a.strip() for a in argnames.split(",")]
        else:
            names = list(argnames)

        def decorator(func):
            cases = []
            for values in argvalues:
                if not isinstance(values, tuple):
                    values = (values,)
                kwargs = dict(zip(names, values))
                cases.append((func.__name__, kwargs))
            func._param_cases = cases
            return func

        return decorator

    def skip(self, reason=""):
        def decorator(func):
            func._skip = reason
            return func

        return decorator


mark = _Mark()


def fixture(func=None, **kwargs):
    if func is None:
        return lambda f: fixture(f, **kwargs)
    func._fixture = True
    return func


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class _Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []

    def record(self, label, ok, detail=""):
        if ok is None:
            self.skipped += 1
            print(f"  SKIP {label}")
        elif ok:
            self.passed += 1
            print(f"  PASS {label}")
        else:
            self.failed += 1
            self.failures.append((label, detail))
            print(f"  FAIL {label}\n{detail}")


def _make_fixtures(func, cache):
    kwargs = {}
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if name in cache:
            kwargs[name] = cache[name]
    return kwargs


def run_module(module_name):
    mod = importlib.import_module(module_name)
    results = _Results()
    fixture_cache = {}

    def get_fixture(name):
        if name in fixture_cache:
            return fixture_cache[name]
        func = getattr(mod, name, None)
        if func is None or not getattr(func, "_fixture", False):
            raise KeyError(name)
        value = func()
        fixture_cache[name] = value
        return value

    for attr_name in sorted(dir(mod)):
        obj = getattr(mod, attr_name)
        if inspect.isclass(obj) and attr_name.startswith("Test"):
            instance = obj()
            setup = getattr(instance, "setup_method", None) or getattr(
                instance, "setUp", None
            )
            teardown = getattr(instance, "teardown_method", None) or getattr(
                instance, "tearDown", None
            )
            for meth_name in sorted(dir(instance)):
                meth = getattr(instance, meth_name)
                if not meth_name.startswith("test_") or not callable(meth):
                    continue
                label = f"{attr_name}.{meth_name}"
                cases = getattr(meth, "_param_cases", [(meth_name, {})])
                if getattr(meth, "_skip", None):
                    results.record(label, None)
                    continue
                for case_name, kwargs in cases:
                    full = label if not kwargs else f"{label}[{kwargs}]"
                    try:
                        sig = inspect.signature(meth)
                        call_kwargs = dict(kwargs)
                        for pname in sig.parameters:
                            if pname not in call_kwargs:
                                call_kwargs[pname] = get_fixture(pname)
                        if setup is not None:
                            try:
                                setup(meth)
                            except TypeError:
                                setup()
                        meth(**call_kwargs)
                        if teardown is not None:
                            try:
                                teardown(meth)
                            except TypeError:
                                teardown()
                        results.record(full, True)
                    except unittest.SkipTest:
                        results.record(full, None)
                    except Exception:
                        results.record(full, False, traceback.format_exc())
        elif inspect.isfunction(obj) and attr_name.startswith("test_"):
            cases = getattr(obj, "_param_cases", [(attr_name, {})])
            if getattr(obj, "_skip", None):
                results.record(attr_name, None)
                continue
            for case_name, kwargs in cases:
                full = attr_name if not kwargs else f"{attr_name}[{kwargs}]"
                try:
                    sig = inspect.signature(obj)
                    call_kwargs = dict(kwargs)
                    for pname in sig.parameters:
                        if pname not in call_kwargs:
                            call_kwargs[pname] = get_fixture(pname)
                    obj(**call_kwargs)
                    results.record(full, True)
                except unittest.SkipTest:
                    results.record(full, None)
                except Exception:
                    results.record(full, False, traceback.format_exc())

    print(
        f"\n{module_name}: {results.passed} passed, "
        f"{results.failed} failed, {results.skipped} skipped"
    )
    return results


if __name__ == "__main__":
    total = _Results()
    for name in sys.argv[1:]:
        r = run_module(name)
        total.passed += r.passed
        total.failed += r.failed
        total.skipped += r.skipped
        total.failures.extend(r.failures)
    print(f"\nTOTAL: {total.passed} passed, {total.failed} failed, {total.skipped} skipped")
    sys.exit(1 if total.failed else 0)
