"""Per-query cost accounting: tokens, latency, sequential calls, $ or GPU-seconds."""
from __future__ import annotations

from dataclasses import dataclass, field

from aex.common.llm import Completion

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {}


@dataclass
class _Entry:
    completion: Completion
    local: bool
    parallel_group: str | None


@dataclass
class Ledger:
    _entries: list[_Entry] = field(default_factory=list)

    def record(self, c: Completion, *, local: bool, parallel_group: str | None = None) -> None:
        self._entries.append(_Entry(c, local, parallel_group))

    @property
    def input_tokens(self) -> int:
        return sum(e.completion.input_tokens for e in self._entries)

    @property
    def output_tokens(self) -> int:
        return sum(e.completion.output_tokens for e in self._entries)

    @property
    def latency_s(self) -> float:
        return sum(e.completion.latency_s for e in self._entries)

    @property
    def sequential_calls(self) -> int:
        ungrouped = sum(1 for e in self._entries if e.parallel_group is None)
        groups = {e.parallel_group for e in self._entries if e.parallel_group is not None}
        return ungrouped + len(groups)

    @property
    def gpu_s(self) -> float | None:
        local = [e.completion.latency_s for e in self._entries if e.local]
        return sum(local) if local else None

    def cost_usd(self, prices: dict[str, tuple[float, float]]) -> float | None:
        api = [e.completion for e in self._entries if not e.local]
        if not api:
            return None
        total = 0.0
        for comp in api:
            if comp.model not in prices:
                raise KeyError(f"no price configured for API model {comp.model!r}")
            per_in, per_out = prices[comp.model]
            total += comp.input_tokens / 1e6 * per_in + comp.output_tokens / 1e6 * per_out
        return total
