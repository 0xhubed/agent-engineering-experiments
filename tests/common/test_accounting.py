import pytest

from aex.common.accounting import Ledger
from aex.common.llm import Completion


def c(model="m", inp=100, out=10, lat=1.0):
    return Completion(text="", input_tokens=inp, output_tokens=out, latency_s=lat, model=model)


def test_sums_tokens_and_latency():
    led = Ledger()
    led.record(c(inp=100, out=10, lat=1.0), local=True)
    led.record(c(inp=50, out=5, lat=0.5), local=True)
    assert (led.input_tokens, led.output_tokens) == (150, 15)
    assert led.latency_s == pytest.approx(1.5)
    assert led.sequential_calls == 2
    assert led.gpu_s == pytest.approx(1.5)
    assert led.cost_usd({}) is None


def test_parallel_group_counts_once():
    led = Ledger()
    led.record(c(), local=True, parallel_group="fanout-1")
    led.record(c(), local=True, parallel_group="fanout-1")
    led.record(c(), local=True)
    assert led.sequential_calls == 2


def test_api_cost_uses_price_table():
    led = Ledger()
    led.record(c(model="frontier", inp=1_000_000, out=500_000), local=False)
    assert led.cost_usd({"frontier": (3.0, 15.0)}) == pytest.approx(3.0 + 7.5)
    assert led.gpu_s is None


def test_missing_price_raises_with_model_name():
    led = Ledger()
    led.record(c(model="unpriced"), local=False)
    with pytest.raises(KeyError, match="unpriced"):
        led.cost_usd({})
