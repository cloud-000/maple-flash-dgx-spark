"""System One discriminative evaluator: single-forward pass logit extraction & scoring."""

from __future__ import annotations

import math
from typing import Any

import torch

from maple_run.system_one.prompt import (
    build_choice_prompt,
    build_noul_prompt,
    build_score_prompt,
)
from maple_run.system_one.schema import (
    ChoiceQuestion,
    ChoiceResult,
    NoulQuestion,
    NoulResult,
    Question,
    ScoreQuestion,
    ScoreResult,
    SystemOneRequest,
    SystemOneResponse,
)


def _safe_softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Compute numerically stable softmax over a list of logits."""
    t = max(float(temperature), 1e-4)
    scaled = [x / t for x in logits]
    max_val = max(scaled)
    exps = [math.exp(x - max_val) for x in scaled]
    total = sum(exps)
    if total <= 0.0:
        n = len(logits)
        return [1.0 / n] * n
    return [e / total for e in exps]


def _resolve_token_id(tokenizer: Any, word: str) -> int | None:
    """Find the primary token ID for a single word or label."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if isinstance(ids, list) and ids:
            return ids[0]
        if torch.is_tensor(ids) and ids.numel() > 0:
            return int(ids.reshape(-1)[0].item())
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        if word in vocab:
            return vocab[word]
    return None


class SystemOneEvaluator:
    """Evaluates System One questions in a single forward pass without autoregressive decode."""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device | None = None):
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = device
        elif hasattr(model, "device"):
            self.device = model.device
        else:
            self.device = torch.device("cpu")

        # Cache common token IDs
        self._pos_tokens = self._find_tokens(["Yes", "yes", "True", "true"])
        self._neg_tokens = self._find_tokens(["No", "no", "False", "false"])

    def _find_tokens(self, words: list[str]) -> list[int]:
        ids: list[int] = []
        for w in words:
            for cand in (w, " " + w):
                tid = _resolve_token_id(self.tokenizer, cand)
                if tid is not None and tid not in ids:
                    ids.append(tid)
        return ids

    def _forward_last_logits(self, prompt: str) -> tuple[torch.Tensor, int]:
        """Run a single forward pass with cache=None and return last token logits."""
        if hasattr(self.tokenizer, "encode"):
            raw = self.tokenizer.encode(prompt, add_special_tokens=False)
            if torch.is_tensor(raw):
                input_ids = raw.to(self.device).long()
            else:
                input_ids = torch.tensor(raw, device=self.device, dtype=torch.long)
        else:
            raise ValueError("Tokenizer must have an 'encode' method")

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        n_tokens = int(input_ids.shape[-1])

        with torch.inference_mode():
            out = self.model.forward(input_ids, cache=None, logits_to_keep=1)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

        # Shape: (1, 1, vocab_size) or (1, vocab_size) -> (vocab_size,)
        last_logits = logits.squeeze().float()
        return last_logits, n_tokens

    def evaluate_noul(
        self, state: Any, question: NoulQuestion, temperature: float = 1.0
    ) -> tuple[NoulResult, int]:
        prompt = build_noul_prompt(state, question)
        logits, n_tokens = self._forward_last_logits(prompt)

        # Extract maximum logit among positive and negative candidate tokens
        pos_logits = [logits[tid].item() for tid in self._pos_tokens if tid < logits.shape[0]]
        neg_logits = [logits[tid].item() for tid in self._neg_tokens if tid < logits.shape[0]]

        if not pos_logits or not neg_logits:
            return NoulResult(noul=0.5), n_tokens

        max_pos = max(pos_logits)
        max_neg = max(neg_logits)

        probs = _safe_softmax([max_neg, max_pos], temperature=temperature)
        prob_true = probs[1]

        return NoulResult(noul=prob_true), n_tokens

    def evaluate_choice(
        self, state: Any, question: ChoiceQuestion, temperature: float = 1.0
    ) -> tuple[ChoiceResult, int]:
        prompt, labels, key_to_label = build_choice_prompt(state, question)
        logits, n_tokens = self._forward_last_logits(prompt)

        # Look up logits for each label (e.g. A, B, C)
        label_logits: list[float] = []
        valid_labels: list[str] = []

        for lbl in labels:
            tid = _resolve_token_id(self.tokenizer, lbl)
            if tid is None:
                tid = _resolve_token_id(self.tokenizer, " " + lbl)
            if tid is not None and tid < logits.shape[0]:
                label_logits.append(logits[tid].item())
                valid_labels.append(lbl)
            else:
                label_logits.append(-1e9)
                valid_labels.append(lbl)

        probs = _safe_softmax(label_logits, temperature=temperature)

        label_to_prob = dict(zip(valid_labels, probs))
        probabilities: dict[str, float] = {}

        for key, lbl in key_to_label.items():
            probabilities[key] = label_to_prob.get(lbl, 0.0)

        # Winning option is argmax
        best_key = max(probabilities.keys(), key=lambda k: probabilities[k])
        confidence = probabilities[best_key]

        return (
            ChoiceResult(
                choice=best_key,
                confidence=confidence,
                probabilities=probabilities,
            ),
            n_tokens,
        )

    def evaluate_score(
        self, state: Any, question: ScoreQuestion, temperature: float = 1.0
    ) -> tuple[ScoreResult, int]:
        prompt, labels = build_score_prompt(state, question)
        logits, n_tokens = self._forward_last_logits(prompt)

        score_logits: list[float] = []
        for lbl in labels:
            tid = _resolve_token_id(self.tokenizer, lbl)
            if tid is None:
                tid = _resolve_token_id(self.tokenizer, " " + lbl)
            if tid is not None and tid < logits.shape[0]:
                score_logits.append(logits[tid].item())
            else:
                score_logits.append(-1e9)

        probs = _safe_softmax(score_logits, temperature=temperature)

        # Expected score: sum(k * p_k) where k starts at 1
        expected_score = sum((idx + 1) * p for idx, p in enumerate(probs))
        confidence = max(probs)

        return (
            ScoreResult(
                score=expected_score,
                confidence=confidence,
                probabilities=probs,
                legend=list(question.criteria),
            ),
            n_tokens,
        )

    def evaluate_request(self, request: SystemOneRequest) -> SystemOneResponse:
        """Evaluate an entire SystemOneRequest across all question primitives."""
        nouls: dict[str, NoulResult] = {}
        choices: dict[str, ChoiceResult] = {}
        scores: dict[str, ScoreResult] = {}
        total_tokens = 0

        for name, q in request.questions.items():
            if isinstance(q, NoulQuestion):
                res, toks = self.evaluate_noul(
                    request.state, q, temperature=request.temperature
                )
                nouls[name] = res
                total_tokens += toks
            elif isinstance(q, ChoiceQuestion):
                res, toks = self.evaluate_choice(
                    request.state, q, temperature=request.temperature
                )
                choices[name] = res
                total_tokens += toks
            elif isinstance(q, ScoreQuestion):
                res, toks = self.evaluate_score(
                    request.state, q, temperature=request.temperature
                )
                scores[name] = res
                total_tokens += toks

        return SystemOneResponse(
            model=request.model,
            nouls=nouls,
            choices=choices,
            scores=scores,
            usage={"input_tokens": total_tokens, "output_tokens": 0},
        )
