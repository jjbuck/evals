"""examples/optimization/model_routing.py: routing and pricing offline, the joint Pareto goal, and the price check."""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "optimization"))
import model_routing as mr  # noqa: E402

ARGS = types.SimpleNamespace(benchmark="toy", repetitions=1, concurrency=1)


def test_selfcheck_routes_once_and_prices_the_conversation():
    mr.selfcheck()
    assert mr.unpriced(), "selfcheck left its test prices in POOL"


def test_a_study_with_an_unset_price_refuses_to_start():
    with pytest.raises(SystemExit, match="set these prices"):
        mr.make_study(ARGS)


def test_the_goal_is_one_joint_pareto_search_and_the_decider_needs_an_endpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for prices in [m.prices for m in mr.POOL.values()] + [mr.ROUTER_LLM_PRICES]:
        for key in prices:
            monkeypatch.setitem(prices, key, 1.0)
    study = mr.make_study(ARGS)
    assert study.projection.primary_and_cost() is None  # no reward-then-cost phases
    assert list(mr.ROUTERS) == ["none", "llm"]  # ROUTING_DECIDER_ENDPOINT is not set in the tests
