"""In-process register machine used by the default tool-task path."""

from __future__ import annotations

from typing import Any


def values_equal(got, want) -> bool:
    if want is None:
        return False
    try:
        return int(got) == int(want)
    except (TypeError, ValueError):
        return got == want


class RegisterMachine:
    """Mutable registers for one episode. Same behavior as the original ToolEnv."""

    def __init__(
        self,
        state: dict[str, Any],
        target: dict[str, Any],
        inject_error: bool = False,
    ):
        self.state = state
        self.target = target
        self.inject_error = inject_error
        self._error_used = False
        self.submitted = False

    def execute(self, name: str, args: dict | None) -> tuple[str, bool]:
        if args is None:
            return "ERROR: arguments must be JSON object", False
        if self.inject_error and not self._error_used:
            self._error_used = True
            return "ERROR: invalid argument type; retry with the schema", False
        try:
            if name == "write":
                key = str(args["key"])
                self.state[key] = args["value"]
                return f"ok write {key}={self.state[key]!r}", True
            if name == "read":
                key = str(args["key"])
                return f"{key}={self.state.get(key)!r}", True
            if name == "add":
                key = str(args["key"])
                delta = int(args["delta"])
                cur = int(self.state.get(key, 0))
                self.state[key] = cur + delta
                return f"ok add {key}={self.state[key]}", True
            if name == "submit":
                key = str(args["key"])
                got = self.state.get(key)
                want = self.target.get(key, next(iter(self.target.values()), None))
                ok = values_equal(got, want)
                self.submitted = True
                return ("PASS" if ok else f"FAIL got={got!r} want={want!r}"), True
        except (KeyError, TypeError, ValueError) as exc:
            return f"ERROR: {exc}", False
        return f"ERROR: unknown tool {name}", False

    def verifier_ok(self) -> bool:
        return all(
            values_equal(self.state.get(key), value) for key, value in self.target.items()
        )
