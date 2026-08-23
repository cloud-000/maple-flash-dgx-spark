"""Single-turn and multi-step tool rollouts for EGGROLL fitness.

The engine owns generation and tool-call orchestration. Environments own
prompts, tools, episode state, and evaluation. ES only sees EpisodeResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from maple_run.eggroll.envs.base import Episode
from maple_run.eggroll.envs.machine import RegisterMachine as ToolEnv
from maple_run.eggroll.rewards import doom_score, ngram_repetition, tool_call_key
from maple_run.eggroll.tasks import Task
from maple_run.eggroll.tools import ToolContext, as_tools, invoke_tool, tool_schemas
from maple_run.generate import (
    CompletionDecoder,
    GenerateResult,
    batched_generate,
    encode_messages,
    encode_user_prompt,
    prompt_in_think,
    run_generate,
    skip_decode_ids,
    stop_token_ids,
)

GenerateFn = Callable[..., GenerateResult]

# Collapse threshold: stop the tool loop instead of decoding a 64k spiral.
_DOOM_NGRAM_STOP = 0.5


def _tool_names(calls: list[dict] | None) -> list[str]:
    names: list[str] = []
    for call in calls or []:
        fn = call.get("function") or {}
        names.append(str(fn.get("name") or call.get("name") or ""))
    return names


def _turn_trace(round_i: int, result: GenerateResult) -> dict[str, Any]:
    return {
        "round": round_i,
        "n_new": result.n_new,
        "prompt_len": result.prompt_len,
        "prefill_s": result.prefill_s,
        "decode_s": result.decode_s,
        "finish_reason": result.finish_reason,
        "tool_names": _tool_names(result.tool_calls),
    }


def _from_generate(result: GenerateResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "reasoning": result.reasoning,
        "n_new": result.n_new,
        "tool_calls": len(result.tool_calls),
        "prompt_len": result.prompt_len,
        "prefill_s": result.prefill_s,
        "decode_s": result.decode_s,
        "finish_reason": result.finish_reason,
    }


def task_token_cap(task: Task, max_tokens: int) -> int:
    cap = task.env.max_new(task.inst)
    return int(cap if cap is not None else max_tokens)


def needs_agent_loop(task: Task) -> bool:
    """True when the next prompt depends on a tool observation."""
    return bool(as_tools(task.env.tools(task.inst)))


def _hit_cap(result: GenerateResult, cap: int) -> bool:
    return result.finish_reason == "length" or (
        result.n_new >= cap and result.finish_reason != "stop"
    )


def _opening(task: Task) -> str | list[dict]:
    msgs = task.env.messages(task.inst)
    if msgs:
        return [dict(m) for m in msgs]
    return task.env.prompt(task.inst)


def score_single_turn(
    task: Task,
    result: GenerateResult,
    cap: int,
    *,
    state: Any = None,
) -> EpisodeResult:
    """Fitness for a completed single-turn generate. Env ``reset`` already ran."""
    doom = doom_score(
        result.text,
        reasoning=result.reasoning,
        tool_calls=result.tool_calls,
        hit_max_tokens=_hit_cap(result, cap),
        n_new=result.n_new,
    )
    episode = Episode(
        inst=task.inst,
        state=state,
        texts=[result.text],
        reasoning=result.reasoning,
        tool_calls=list(result.tool_calls),
        n_new=result.n_new,
        doom=doom,
    )
    verify, parts = task.env.verify(task.inst, episode)
    parts = {"doom": doom, **parts}
    fit = task.env.combine(verify, doom)
    return EpisodeResult(
        fitness=float(task.weight) * fit,
        kind=task.kind,
        parts=parts,
        task=task,
        **_from_generate(result),
    )


def _results_from_batched(
    tokenizer,
    prompt_ids: list,
    batched,
    *,
    skip_ids: set[int],
) -> list[GenerateResult]:
    out: list[GenerateResult] = []
    for i, ids in enumerate(batched.token_ids):
        decoder = CompletionDecoder(
            tokenizer,
            in_think=prompt_in_think(tokenizer, prompt_ids[i]),
            skip_ids=skip_ids,
        )
        kept: list[int] = []
        for tid in ids:
            decoder.push(int(tid))
            kept.append(int(tid))
            if decoder.should_stop:
                break
        reasoning, text = decoder.finalize()
        reason = batched.finish_reasons[i]
        if decoder.tool_calls:
            reason = "tool_calls"
        out.append(
            GenerateResult(
                text=text,
                prompt_len=batched.prompt_lens[i],
                n_new=len(kept),
                prefill_s=batched.prefill_s,
                decode_s=batched.decode_s,
                token_ids=kept,
                finish_reason=reason,
                reasoning=reasoning,
                tool_calls=decoder.tool_calls,
            )
        )
    return out


def run_single_turn_batch(
    model,
    tokenizer,
    tasks: list[Task],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None = None,
) -> list[EpisodeResult]:
    """Decode several same-member single-turn tasks in one packed batch."""
    if not tasks:
        return []
    if len(tasks) == 1:
        return [
            run_task(
                model,
                tokenizer,
                tasks[0],
                max_tokens=max_tokens,
                max_tool_rounds=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )
        ]
    states = [task.env.reset(task.inst) for task in tasks]
    caps = [task_token_cap(task, max_tokens) for task in tasks]
    prompt_ids = []
    for task in tasks:
        msgs = task.env.messages(task.inst)
        if msgs:
            prompt_ids.append(encode_messages(tokenizer, msgs, model.device))
        else:
            prompt_ids.append(
                encode_user_prompt(tokenizer, task.env.prompt(task.inst), model.device)
            )
    stop_ids = stop_token_ids(model, tokenizer)
    batched = batched_generate(
        model,
        prompt_ids,
        max_tokens=max(caps),
        max_new=caps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        stop_ids=stop_ids,
        graphs=False,
        log=lambda _m: None,
    )
    results = _results_from_batched(
        tokenizer,
        prompt_ids,
        batched,
        skip_ids=skip_decode_ids(tokenizer, extra=stop_ids),
    )
    scored = [
        score_single_turn(task, result, cap, state=state)
        for task, result, cap, state in zip(tasks, results, caps, states, strict=True)
    ]
    # Packed GEMM timings are for the whole chunk, not per row.
    batch_id = id(batched)
    for ep in scored:
        ep.batch_id = batch_id
    return scored


@dataclass
class EpisodeResult:
    fitness: float
    kind: str
    parts: dict[str, float] = field(default_factory=dict)
    text: str = ""
    reasoning: str = ""
    n_new: int = 0
    tool_calls: int = 0
    prompt_len: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    finish_reason: str = ""
    rounds: int = 1
    batch_id: int | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    task: Task | None = None


def _run_once(
    model,
    tokenizer,
    prompt: str | list,
    *,
    tools: list | None = None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    generate_fn: GenerateFn | None,
) -> GenerateResult:
    if generate_fn is not None:
        return generate_fn(
            model,
            tokenizer,
            prompt,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
    if tools or not isinstance(prompt, str):
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        input_ids = encode_messages(
            tokenizer, messages, model.device, tools=tools
        )
    else:
        input_ids = encode_user_prompt(tokenizer, prompt, model.device)
    return run_generate(
        model,
        tokenizer,
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        log=None,
        use_graph=False,
    )


def run_task(
    model,
    tokenizer,
    task: Task,
    *,
    max_tokens: int,
    max_tool_rounds: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None = None,
    generate_fn: GenerateFn | None = None,
) -> EpisodeResult:
    cap = task_token_cap(task, max_tokens)
    tools = as_tools(task.env.tools(task.inst))
    if not tools:
        state = task.env.reset(task.inst)
        result = _run_once(
            model,
            tokenizer,
            _opening(task),
            max_tokens=cap,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            generate_fn=generate_fn,
        )
        return score_single_turn(task, result, cap, state=state)
    rounds = task.env.max_rounds(task.inst)
    if rounds is None:
        rounds = max_tool_rounds
    return _run_agent_loop(
        model,
        tokenizer,
        task,
        tools=tools,
        max_tokens=cap,
        max_tool_rounds=rounds,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        generate_fn=generate_fn,
    )


def _run_agent_loop(
    model,
    tokenizer,
    task: Task,
    *,
    tools: list,
    max_tokens: int,
    max_tool_rounds: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    generate_fn: GenerateFn | None,
) -> EpisodeResult:
    env = task.env
    inst = task.inst
    schemas = tool_schemas(tools)
    state = env.reset(inst)
    msgs = env.messages(inst)
    messages: list[dict] = (
        [dict(m) for m in msgs]
        if msgs
        else [{"role": "user", "content": env.prompt(inst)}]
    )
    episode = Episode(inst=inst, messages=messages, state=state)
    ctx = ToolContext(env=env, inst=inst, state=state)
    last_reason = ""
    first: str | list = messages if msgs else env.prompt(inst)
    prompt_len = 0
    prefill_s = 0.0
    decode_s = 0.0
    finish_reason = ""
    turns: list[dict[str, Any]] = []

    for round_i in range(max(1, max_tool_rounds)):
        result = _generate_turn(
            model,
            tokenizer,
            first if round_i == 0 else messages,
            tools=schemas,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=None if seed is None else seed + round_i,
            generate_fn=generate_fn,
            first_turn=round_i == 0,
        )
        episode.texts.append(result.text)
        last_reason = result.reasoning
        episode.reasoning = last_reason
        episode.n_new += result.n_new
        if round_i == 0:
            prompt_len = result.prompt_len
        prefill_s += result.prefill_s
        decode_s += result.decode_s
        finish_reason = result.finish_reason
        turns.append(_turn_trace(round_i, result))
        if ngram_repetition(result.text) >= _DOOM_NGRAM_STOP:
            break
        if not result.tool_calls:
            break
        messages.append(
            {
                "role": "assistant",
                "content": result.text,
                "tool_calls": result.tool_calls,
            }
        )
        for call in result.tool_calls:
            episode.tool_calls.append(call)
            outcome = invoke_tool(tools, call, ctx)
            if outcome.ok:
                episode.valid_calls += 1
                if episode.extra.get("saw_error"):
                    episode.recovered = True
            else:
                episode.invalid_calls += 1
                episode.extra["saw_error"] = True
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": outcome.content,
                }
            )
        episode.messages = messages
        if env.done(state):
            break
        keys = [tool_call_key(c) for c in result.tool_calls]
        if len(keys) >= 2 and len(set(keys)) == 1:
            break

    blob = episode.text
    doom = doom_score(
        blob, reasoning=last_reason, tool_calls=episode.tool_calls, n_new=episode.n_new
    )
    episode.doom = doom
    verify, parts = env.verify(inst, episode)
    parts = {"doom": doom, **parts}
    fit = env.combine(verify, doom)
    return EpisodeResult(
        fitness=float(task.weight) * fit,
        kind=task.kind,
        parts=parts,
        text=blob,
        reasoning=last_reason,
        n_new=episode.n_new,
        tool_calls=len(episode.tool_calls),
        prompt_len=prompt_len,
        prefill_s=prefill_s,
        decode_s=decode_s,
        finish_reason=finish_reason,
        rounds=len(turns),
        turns=turns,
        task=task,
    )


def _generate_turn(
    model,
    tokenizer,
    prompt_or_messages,
    *,
    tools: list | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    generate_fn: GenerateFn | None,
    first_turn: bool,
) -> GenerateResult:
    if generate_fn is not None:
        return generate_fn(
            model,
            tokenizer,
            prompt_or_messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
    if first_turn and isinstance(prompt_or_messages, str):
        input_ids = encode_messages(
            tokenizer,
            [{"role": "user", "content": prompt_or_messages}],
            model.device,
            tools=tools,
        )
    else:
        input_ids = encode_messages(
            tokenizer, prompt_or_messages, model.device, tools=tools
        )
    return run_generate(
        model,
        tokenizer,
        input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        log=None,
        use_graph=False,
    )
