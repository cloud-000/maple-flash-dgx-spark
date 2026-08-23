"""Generic Task, tool protocol, recipes, and ES-agnostic rollout tests."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from maple_run.cli import main
from maple_run.eggroll.envs.base import Episode, SingleTurnEnv, ToolEnv
from maple_run.eggroll.recipes import (
    TrainingRecipe,
    get_recipe,
    list_recipes,
    load_recipe_file,
    register_recipe,
)
from maple_run.eggroll.rollout import needs_agent_loop, run_task
from maple_run.eggroll.tasks import Task
from maple_run.eggroll.tools import (
    ToolContext,
    ToolResult,
    WebSearchTool,
    as_tools,
    dispatch_named,
    function_tool,
    invoke_tool,
    parse_tool_call,
)
from maple_run.eggroll.train import TrainConfig, train
from maple_run.generate import GenerateResult
from maple_run.eggroll_envs import (
    DefaultRecipe,
    DoomEnv,
    ReasonEnv,
    RefusalEnv,
    RegisterEnv,
)


class _EchoEnv(SingleTurnEnv):
    kind = "echo"

    def sample(self, rng) -> dict[str, Any]:
        return {"prompt": "say ok", "needle": "ok"}

    def catalog(self) -> list[Any]:
        return [self.sample(__import__("random").Random(0))]

    def prompt(self, inst: Any) -> str:
        return str(inst["prompt"])

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        hit = 1.0 if inst["needle"] in episode.text else -1.0
        return hit, {"echo": hit}


class _BumpTool:
    name = "bump"

    def schema(self) -> dict:
        return function_tool(
            "bump", "Add n to a counter.", {"n": {"type": "integer"}}, ["n"]
        )

    def call(self, args: dict | None, ctx: ToolContext) -> ToolResult:
        ctx.state["called"] = True
        ctx.state["n"] = int(ctx.state.get("n", 0)) + int((args or {}).get("n", 0))
        return ToolResult(content=f"n={ctx.state['n']}", ok=True)


class _CounterEnv(ToolEnv):
    kind = "counter"
    abstract = False

    def sample(self, rng) -> dict[str, Any]:
        return {"prompt": "bump the counter to 3", "target": 3}

    def catalog(self) -> list[Any]:
        return [self.sample(__import__("random").Random(0))]

    def prompt(self, inst: Any) -> str:
        return str(inst["prompt"])

    def tools(self, inst: Any) -> list:
        return [_BumpTool()]

    def initial_state(self, inst: Any) -> dict[str, Any]:
        return {"n": 0, "called": False}

    def done(self, state: Any = None) -> bool:
        return bool(state and state.get("n", 0) >= 3)

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        n = (episode.state or {}).get("n", 0)
        hit = 1.0 if n == inst["target"] else -1.0
        return hit, {"counter": hit, "n": float(n)}

    def combine(self, verify: float, doom: float) -> float:
        return float(verify)


class _SearchishEnv(ToolEnv):
    """Second env that reuses WebSearchTool without owning a tool loop."""

    kind = "searchish"
    abstract = False

    def sample(self, rng) -> dict[str, Any]:
        return {
            "query": "year?",
            "results": [{"title": "t", "url": "https://example.test/x", "snippet": "1999"}],
        }

    def prompt(self, inst: Any) -> str:
        return str(inst["query"])

    def tools(self, inst: Any) -> list:
        return [WebSearchTool()]

    def initial_state(self, inst: Any) -> dict[str, Any]:
        return {"search_count": 0}

    def search(self, inst: Any, query: str) -> dict[str, Any]:
        return {"results": inst["results"]}

    def verify(self, inst: Any, episode: Episode) -> tuple[float, dict[str, float]]:
        return 1.0 if "1999" in episode.text else 0.0, {}


def _tool_call(name: str, args: dict | str) -> dict:
    body = args if isinstance(args, str) else json.dumps(args)
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": body},
    }


def _scripted(calls_by_round: list[list[dict]], text: str = "ok"):
    turn = {"i": 0}

    def generate_fn(model, tokenizer, prompt, **kwargs):
        i = turn["i"]
        turn["i"] += 1
        calls = calls_by_round[i] if i < len(calls_by_round) else []
        return GenerateResult(
            text=text,
            prompt_len=4,
            n_new=4,
            prefill_s=0.0,
            decode_s=0.01,
            tool_calls=calls,
        )

    return generate_fn


class _FakePacked:
    def __init__(self, n, k):
        self.packed_weight = torch.zeros(n, k // 16, dtype=torch.uint32)
        self.row_alpha = torch.ones(n)


class _FakeLayer:
    def __init__(self):
        self.self_attn = SimpleNamespace(
            qkv_proj=_FakePacked(32, 16),
            o_proj=_FakePacked(16, 16),
        )
        self.mlp = SimpleNamespace(down=_FakePacked(16, 16), up_gate=_FakePacked(16, 16))


class _FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32
    layers = [_FakeLayer()]
    lm_head = None
    eggroll = None


def test_task_is_generic_and_frozen():
    env = RefusalEnv()
    inst = {"prompt": "hi", "should_refuse": False}
    task = env.bind(inst)
    assert set(Task.__dataclass_fields__) == {"env", "inst", "meta", "weight"}
    assert task.env is env
    assert task.inst == inst
    assert task.inst is not inst
    inst["prompt"] = "mutated"
    assert task.inst["prompt"] == "hi"
    assert task.meta["should_refuse"] is False
    with pytest.raises(FrozenInstanceError):
        task.weight = 2.0
    with pytest.raises(TypeError):
        task.meta["x"] = 1


def test_generic_single_turn_rollout():
    env = _EchoEnv()
    task = env.bind(env.catalog()[0])
    assert not needs_agent_loop(task)

    def generate_fn(model, tokenizer, prompt, **kwargs):
        assert kwargs.get("tools") in (None, [])
        return GenerateResult(
            text="ok grove", prompt_len=2, n_new=2, prefill_s=0.0, decode_s=0.01
        )

    result = run_task(
        object(),
        object(),
        task,
        max_tokens=16,
        max_tool_rounds=4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        generate_fn=generate_fn,
    )
    assert result.kind == "echo"
    assert result.parts["echo"] == 1.0
    assert result.fitness > 0
    assert result.rounds == 1


def test_generic_tool_loop_and_malformed_calls():
    env = _CounterEnv()
    task = env.bind(env.catalog()[0])
    assert needs_agent_loop(task)

    kwargs = dict(
        max_tokens=32,
        max_tool_rounds=6,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
    seen_schemas: list[Any] = []
    scripted = _scripted(
        [
            [_tool_call("bump", {"n": 1})],
            [_tool_call("bump", {"n": 2})],
        ]
    )

    def generate_fn(model, tokenizer, prompt, **kw):
        seen_schemas.append(kw.get("tools"))
        return scripted(model, tokenizer, prompt, **kw)

    good = run_task(
        object(),
        object(),
        task,
        generate_fn=generate_fn,
        **kwargs,
    )
    assert seen_schemas and seen_schemas[0]
    assert seen_schemas[0][0]["function"]["name"] == "bump"
    assert good.parts["n"] == 3.0
    assert good.parts["counter"] == 1.0
    assert good.rounds >= 2

    env2 = _CounterEnv()
    task2 = env2.bind(env2.catalog()[0])
    bad = run_task(
        object(),
        object(),
        task2,
        generate_fn=_scripted(
            [
                [_tool_call("bump", "not-json")],
                [_tool_call("missing", {"n": 1})],
                [_tool_call("bump", {})],
                [_tool_call("bump", {"n": 3})],
            ]
        ),
        **kwargs,
    )
    assert task2.env._state["called"] is True
    assert bad.task.env._state["n"] == 3
    assert bad.parts.get("doom") is not None

    limited = run_task(
        object(),
        object(),
        _CounterEnv().bind(_CounterEnv().catalog()[0]),
        generate_fn=_scripted([[_tool_call("bump", {"n": 1})]] * 8),
        max_tokens=32,
        max_tool_rounds=2,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
    assert limited.rounds == 2
    assert limited.task.env._state["n"] == 2


def test_reusable_tools_shared_across_environments():
    a = _SearchishEnv()
    b = _SearchishEnv()
    inst_a = a.sample(__import__("random").Random(0))
    inst_b = b.sample(__import__("random").Random(1))
    state_a = a.reset(inst_a)
    state_b = b.reset(inst_b)
    tool_a = as_tools(a.tools(inst_a))[0]
    tool_b = as_tools(b.tools(inst_b))[0]
    assert type(tool_a) is type(tool_b) is WebSearchTool
    first, ok = a.step("web_search", {"query": "year?"})
    assert ok and "1999" in first
    again, ok2 = a.step("web_search", {"query": "again"})
    assert not ok2 and "already searched" in again
    other, ok3 = b.step("web_search", {"query": "year?"})
    assert ok3 and "1999" in other
    ctx = ToolContext(env=a, inst=inst_a, state=state_a)
    malformed = invoke_tool(
        as_tools(a.tools(inst_a)), _tool_call("web_search", "nope"), ctx
    )
    assert malformed.malformed and not malformed.ok
    name, args, err = parse_tool_call(_tool_call("web_search", {"query": "x"}))
    assert name == "web_search" and args == {"query": "x"} and err is None
    missing = dispatch_named(
        as_tools(a.tools(inst_a)), "web_search", {}, ToolContext(env=a, inst=inst_a, state=state_b)
    )
    assert not missing.ok and "missing required" in missing.content


def test_recipe_registration_and_cli_selection(capsys):
    names = list_recipes()
    assert "default" in names
    assert "agent" in names
    assert "ipi" in names
    default = get_recipe("default")
    kinds = {e.kind for e in default.environments()}
    assert kinds == {"refusal", "doom", "reason", "tool"}
    agent = get_recipe("agent")
    assert agent.defaults()["max_tokens"] == 512
    assert agent.defaults()["max_tool_rounds"] == 8
    kept, _ = agent.resolve(SimpleNamespace(max_tokens=256), set())
    assert kept["max_tokens"] == 512
    bumped, ov = agent.resolve(SimpleNamespace(max_tokens=128), {"max_tokens"})
    assert bumped["max_tokens"] == 128
    assert ov == {"max_tokens": 128}
    ns = SimpleNamespace(population=32, steps=None, lr=None)
    resolved, overrides = default.resolve(ns, {"population"})
    assert resolved["population"] == 32
    assert overrides == {"population": 32}
    assert resolved["steps"] == default.defaults()["steps"]

    @register_recipe
    class _TinyRecipe(TrainingRecipe):
        name = "tiny_test_recipe"

        def environments(self):
            return [RefusalEnv()]

    assert get_recipe("tiny_test_recipe").environments()[0].kind == "refusal"

    with pytest.raises(SystemExit) as exc:
        main(["eggroll", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--recipe" in out
    with pytest.raises(SystemExit) as err:
        main(["eggroll", "--model", "x", "--output", "y", "--recipe", "no-such-recipe"])
    assert err.value.code == 2


def test_resolved_config_records_recipe_defaults_after_cli_overrides(tmp_path: Path):
    model = _FakeModel()
    recipe = DefaultRecipe()
    defaults = recipe.defaults()
    cfg = TrainConfig(
        model_dir="unused",
        output_dir=str(tmp_path),
        steps=1,
        population=2,
        prompts_per_step=1,
        modules=("qkv",),
        recipe_name="default",
        recipe_defaults=defaults,
        cli_overrides={"population": 2},
        log=lambda *_: None,
    )

    def generate_fn(model, tokenizer, prompt, **kwargs):
        return GenerateResult(
            text="Ada Lovelace wrote notes.",
            prompt_len=4,
            n_new=4,
            prefill_s=0.0,
            decode_s=0.01,
        )

    train(
        cfg,
        model=model,
        tokenizer=object(),
        suite=[RefusalEnv().bind(RefusalEnv().catalog()[0])],
        generate_fn=generate_fn,
    )
    blob = json.loads((tmp_path / "config.json").read_text())
    assert set(blob) >= {"recipe", "environment", "rollout", "optimizer", "cli"}
    assert blob["recipe"]["name"] == "default"
    assert blob["recipe"]["defaults"]["population"] == defaults["population"]
    assert blob["recipe"]["dirs"] == []
    assert blob["optimizer"]["population"] == 2
    assert blob["cli"]["overrides"]["population"] == 2
    assert blob["cli"]["recipe"] == "default"
    assert blob["rollout"]["max_tokens"] == defaults["max_tokens"]
    assert "RefusalEnv" in blob["recipe"]["environments"]


def test_es_loop_is_agnostic_to_environment(tmp_path: Path):
    model = _FakeModel()
    env = _EchoEnv()
    suite = [env.bind(env.catalog()[0])]

    def generate_fn(model, tokenizer, prompt, **kwargs):
        sign = getattr(model.eggroll, "_sign", 1.0)
        text = "ok" if sign > 0 else "nope"
        return GenerateResult(
            text=text, prompt_len=2, n_new=2, prefill_s=0.0, decode_s=0.01
        )

    history = train(
        TrainConfig(
            model_dir="unused",
            output_dir=str(tmp_path),
            steps=1,
            population=2,
            prompts_per_step=1,
            modules=("qkv",),
            log=lambda *_: None,
        ),
        model=model,
        tokenizer=object(),
        suite=suite,
        generate_fn=generate_fn,
    )
    assert "echo" in history[0].by_kind
    assert history[0].pair_fitness[0][0] != history[0].pair_fitness[0][1]
    assert model.layers[0].self_attn.qkv_proj.eggroll.residual_rank == 1
    rec = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    assert rec["kind"] == "echo"
    assert rec["env"] == "_EchoEnv"


def test_recipe_plugin_file(tmp_path: Path):
    plugin = tmp_path / "file_recipe.py"
    plugin.write_text(
        """
from maple_run.eggroll.recipes import TrainingRecipe, register_recipe
from maple_run.eggroll_envs import RefusalEnv

@register_recipe
class FileRecipe(TrainingRecipe):
    name = "from_file"

    def defaults(self):
        out = super().defaults()
        out["steps"] = 3
        return out

    def environments(self):
        return [RefusalEnv()]
"""
    )
    load_recipe_file(plugin)
    recipe = get_recipe("from_file")
    assert recipe.defaults()["steps"] == 3
    assert recipe.environments()[0].kind == "refusal"


def test_builtin_envs_delegate_domain_fields():
    assert RefusalEnv().meta(RefusalEnv().catalog()[0])["should_refuse"] in {True, False}
    assert "gold" in ReasonEnv().meta(ReasonEnv().catalog()[0])
    assert DoomEnv().tools(DoomEnv().catalog()[0]) in (None, [])
    reg = RegisterEnv()
    inst = reg.catalog()[-1]
    assert inst["inject_error"] is True
    assert [t.name for t in reg.tools(inst)] == ["write", "read", "add", "submit"]
