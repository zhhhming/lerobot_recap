#!/usr/bin/env python

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

import lerobot.policies.pi0.modeling_pi0 as pi0_modeling
import lerobot.policies.pi05.modeling_pi05 as pi05_modeling


POLICY_CASES = (
    pytest.param(pi0_modeling.PI0Pytorch, pi0_modeling.PI0Policy, True, id="pi0"),
    pytest.param(pi05_modeling.PI05Pytorch, pi05_modeling.PI05Policy, False, id="pi05"),
)


class _AttentionBackbone(nn.Module):
    """Small differentiable backbone that obeys the model-provided attention mask."""

    def __init__(self, hidden_dim: int = 8, vocab_size: int = 16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.last_allowed_attention = None

        q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        language_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=q_proj))]
        )
        self.paligemma = SimpleNamespace(
            lm_head=self.lm_head,
            model=SimpleNamespace(language_model=language_model),
        )

    def embed_language_tokens(self, token_ids):
        return self.token_embedding(token_ids)

    def forward(
        self,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        adarms_cond=None,
    ):
        del position_ids, past_key_values, use_cache, adarms_cond
        prefix_embs, suffix_embs = inputs_embeds
        prefix_len = prefix_embs.shape[1]
        full_embs = torch.cat([prefix_embs, suffix_embs], dim=1)

        allowed = attention_mask[:, 0].eq(0.0)
        self.last_allowed_attention = allowed.detach().clone()
        weights = allowed.to(dtype=full_embs.dtype)
        context = torch.bmm(weights, full_embs) / weights.sum(dim=-1, keepdim=True).clamp(min=1)
        outputs = full_embs + context
        return [outputs[:, :prefix_len], outputs[:, prefix_len:]], None


class _TrainingHarness(nn.Module):
    def __init__(self, core_cls, *, is_pi0: bool, subtask_dropout_prob: float):
        super().__init__()
        self.is_pi0 = is_pi0
        self.config = SimpleNamespace(chunk_size=2, subtask_dropout_prob=subtask_dropout_prob)
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert = _AttentionBackbone()
        self.action_in_proj = nn.Linear(3, 8)
        self.action_out_proj = nn.Linear(8, 3)
        if is_pi0:
            self.state_in_proj = nn.Linear(3, 8)

        self.embed_prefix = MethodType(core_cls.embed_prefix, self)
        self._prepare_attention_masks_4d = MethodType(core_cls._prepare_attention_masks_4d, self)
        self.forward = MethodType(core_cls.forward, self)

    def _apply_checkpoint(self, function, *args, **kwargs):
        return function(*args, **kwargs)

    def embed_suffix(self, *args):
        if self.is_pi0:
            state, noisy_actions, _timestep = args
            state_embs = self.state_in_proj(state)[:, None, :]
            action_embs = self.action_in_proj(noisy_actions)
            embs = torch.cat([state_embs, action_embs], dim=1)
            att_masks = torch.tensor([[True, True, False]], device=embs.device)
        else:
            noisy_actions, _timestep = args
            embs = self.action_in_proj(noisy_actions)
            att_masks = torch.tensor([[True, False]], device=embs.device)

        pad_masks = torch.ones(embs.shape[:2], dtype=torch.bool, device=embs.device)
        return embs, pad_masks, att_masks.expand(embs.shape[0], -1), None


def _run_training_forward(core_cls, *, is_pi0, memory_kept, subtask_dropout_prob):
    torch.manual_seed(7)
    model = _TrainingHarness(
        core_cls,
        is_pi0=is_pi0,
        subtask_dropout_prob=subtask_dropout_prob,
    )
    model.train()

    # Index 2 is the distinctive Memory token. Dropping memory masks that fixed
    # main-prompt position without changing the current subtask target segment.
    main_tokens = torch.tensor([[1, 2, 7, 0]])
    main_masks = torch.tensor([[True, True, memory_kept, False]])
    subtask_tokens = torch.tensor([[3, 4, 5]])
    subtask_masks = torch.tensor([[True, True, True]])
    original_subtask_tokens = subtask_tokens.clone()
    original_subtask_masks = subtask_masks.clone()
    actions = torch.tensor([[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]])
    noise = torch.zeros_like(actions)
    time = torch.tensor([0.5])

    args = [[], [], main_tokens, main_masks]
    if is_pi0:
        args.append(torch.tensor([[0.2, 0.4, 0.6]]))
    args.append(actions)

    fm_loss, ce_loss = model.forward(
        *args,
        noise=noise,
        time=time,
        subtask_tokens=subtask_tokens,
        subtask_masks=subtask_masks,
    )
    (fm_loss.mean() + ce_loss.mean()).backward()

    assert torch.equal(subtask_tokens, original_subtask_tokens)
    assert torch.equal(subtask_masks, original_subtask_masks)
    return model, fm_loss, ce_loss


@pytest.mark.parametrize(("core_cls", "policy_cls", "is_pi0"), POLICY_CASES)
@pytest.mark.parametrize("memory_kept", [False, True], ids=["memory_drop", "memory_keep"])
@pytest.mark.parametrize(
    "subtask_dropout_prob",
    [0.0, 1.0],
    ids=["subtask_keep", "subtask_drop"],
)
def test_memory_prefix_is_visible_to_subtask_and_fm_in_all_dropout_combinations(
    core_cls,
    policy_cls,
    is_pi0,
    memory_kept,
    subtask_dropout_prob,
):
    del policy_cls
    model, fm_loss, ce_loss = _run_training_forward(
        core_cls,
        is_pi0=is_pi0,
        memory_kept=memory_kept,
        subtask_dropout_prob=subtask_dropout_prob,
    )

    assert torch.isfinite(fm_loss).all()
    assert torch.isfinite(ce_loss).all()

    allowed = model.paligemma_with_expert.last_allowed_attention[0]
    memory_index = 2
    subtask_start, subtask_end = 4, 7
    suffix_start = subtask_end

    # Current subtask is still causal and sees the main prompt, including Memory
    # exactly when that condition was kept by the processor.
    assert allowed[subtask_start:subtask_end, memory_index].eq(memory_kept).all()
    assert torch.equal(
        allowed[subtask_start:subtask_end, subtask_start:subtask_end],
        torch.tril(torch.ones(3, 3, dtype=torch.bool)),
    )

    # Existing subtask dropout only removes suffix -> current-subtask attention.
    # It must never remove the Memory prefix condition.
    assert allowed[suffix_start:, memory_index].eq(memory_kept).all()
    assert allowed[suffix_start:, subtask_start:subtask_end].eq(subtask_dropout_prob == 0.0).all()

    memory_token_id = 7
    memory_grad = model.paligemma_with_expert.token_embedding.weight.grad[memory_token_id]
    if memory_kept:
        assert memory_grad.abs().sum() > 0
    else:
        assert torch.equal(memory_grad, torch.zeros_like(memory_grad))


@pytest.mark.parametrize(("core_cls", "policy_cls", "is_pi0"), POLICY_CASES)
def test_policy_reset_clears_subtask_semantic_state(core_cls, policy_cls, is_pi0):
    del core_cls, is_pi0
    policy = SimpleNamespace(
        config=SimpleNamespace(n_action_steps=2),
        last_subtask_text="Subtask: stale; Progress: 0.9",
        _last_logged_subtask_text="Subtask: stale; Progress: 0.9",
        model=SimpleNamespace(_last_subtask_tokens=torch.tensor([[1, 2, 3]])),
    )

    policy_cls.reset(policy)

    assert policy.last_subtask_text == ""
    assert policy._last_logged_subtask_text is None
    assert policy.model._last_subtask_tokens is None


class _StateDictBackbone(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        del args, kwargs
        self.projection = nn.Linear(2, 2)


@pytest.mark.parametrize(
    ("module", "core_cls", "is_pi0"),
    (
        pytest.param(pi0_modeling, pi0_modeling.PI0Pytorch, True, id="pi0"),
        pytest.param(pi05_modeling, pi05_modeling.PI05Pytorch, False, id="pi05"),
    ),
)
def test_memory_config_adds_no_model_state_dict_keys(monkeypatch, module, core_cls, is_pi0):
    monkeypatch.setattr(module, "get_gemma_config", lambda _variant: SimpleNamespace(width=8))
    monkeypatch.setattr(module, "PaliGemmaWithExpertModel", _StateDictBackbone)

    def make_config(use_memory_conditioning):
        return SimpleNamespace(
            use_memory_conditioning=use_memory_conditioning,
            paligemma_variant="fake",
            action_expert_variant="fake",
            image_resolution=(8, 8),
            dtype="float32",
            freeze_vision_encoder=False,
            train_expert_only=False,
            max_action_dim=3,
            max_state_dim=3,
            compile_model=False,
        )

    memory_off = core_cls(make_config(False))
    memory_on = core_cls(make_config(True))

    assert memory_on.state_dict().keys() == memory_off.state_dict().keys()
    incompatible = memory_on.load_state_dict(memory_off.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
