#!/usr/bin/env python

"""Relative-action helpers for Real-Time Chunking."""

from __future__ import annotations

import torch

from lerobot.processor import (
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_relative_actions,
)


def reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_step: RelativeActionsProcessorStep,
    normalizer_step: NormalizerProcessorStep | None,
    policy_device: torch.device | str,
) -> torch.Tensor:
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)

    action_cpu = prev_actions_absolute.detach().cpu()
    mask = relative_step._build_mask(action_cpu.shape[-1])
    relative_actions = to_relative_actions(action_cpu, state, mask)

    transition = create_transition(action=relative_actions)
    if normalizer_step is not None:
        transition = normalizer_step(transition)

    return transition[TransitionKey.ACTION].to(policy_device)
