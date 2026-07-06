#!/usr/bin/env python

from types import SimpleNamespace

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, OPENPI_ATTENTION_MASK_VALUE
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def test_build_subtask_seed_attention_mask_is_prefix_visible_and_seed_causal():
    model = SimpleNamespace()
    prefix_pad_masks = torch.tensor([[True, False, True]])

    mask = PI05Pytorch._build_subtask_seed_attention_mask(model, prefix_pad_masks, seed_len=3)

    expected = torch.tensor(
        [
            [
                [True, False, True, True, False, False],
                [True, False, True, True, True, False],
                [True, False, True, True, True, True],
            ]
        ]
    )
    assert torch.equal(mask, expected)


class _IdentityLmHead:
    def __call__(self, hidden):
        return hidden


class _FakePaliGemmaWithExpert:
    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        self.forward_calls = []
        self.paligemma = SimpleNamespace(
            lm_head=_IdentityLmHead(),
            model=SimpleNamespace(
                language_model=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
            ),
        )

    def embed_language_tokens(self, token_ids):
        return torch.nn.functional.one_hot(token_ids, num_classes=self.vocab_size).float()

    def forward(
        self,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        adarms_cond=None,
    ):
        self.forward_calls.append(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "use_cache": use_cache,
            }
        )
        token_ids = inputs_embeds[0].argmax(dim=-1)
        next_ids = torch.where(
            token_ids[:, -1] == 4,
            torch.full_like(token_ids[:, -1], 5),
            torch.full_like(token_ids[:, -1], 4),
        )
        hidden = torch.nn.functional.one_hot(next_ids[:, None], num_classes=self.vocab_size).float()
        return (hidden, None), object()


def _make_subtask_generation_model(max_decode_tokens=4):
    model = SimpleNamespace(
        config=SimpleNamespace(
            subtask_max_decode_tokens=max_decode_tokens,
            subtask_decode_temperature=0.0,
        ),
        paligemma_with_expert=_FakePaliGemmaWithExpert(),
    )
    model._prepare_attention_masks_4d = PI05Pytorch._prepare_attention_masks_4d.__get__(model)
    model._embed_language_generation_tokens = PI05Pytorch._embed_language_generation_tokens.__get__(model)
    model._sample_subtask_token = PI05Pytorch._sample_subtask_token.__get__(model)
    model._build_subtask_seed_attention_mask = PI05Pytorch._build_subtask_seed_attention_mask.__get__(model)
    model._generate_subtask = PI05Pytorch._generate_subtask.__get__(model)
    return model


def test_generate_subtask_extends_cache_and_stops_after_eos():
    model = _make_subtask_generation_model()
    prefix_pad_masks = torch.tensor([[True, False, True]])
    prefix_reference_embs = torch.zeros(1, 3, 8)

    tokens, extended_masks, _ = model._generate_subtask(
        past_key_values=object(),
        prefix_pad_masks=prefix_pad_masks,
        seed_token_ids=torch.tensor([1, 2]),
        eos_token_id=5,
        prefix_reference_embs=prefix_reference_embs,
    )

    assert torch.equal(tokens, torch.tensor([[1, 2, 4, 5]]))
    assert torch.equal(extended_masks, torch.tensor([[True, False, True, True, True, True, True]]))
    assert len(model.paligemma_with_expert.forward_calls) == 3

    seed_call = model.paligemma_with_expert.forward_calls[0]
    seed_mask = seed_call["attention_mask"].eq(0.0)
    assert torch.equal(
        seed_mask,
        torch.tensor([[[[True, False, True, True, False], [True, False, True, True, True]]]]),
    )
    assert torch.equal(seed_call["position_ids"], torch.tensor([[2, 3]]))

    eos_call = model.paligemma_with_expert.forward_calls[-1]
    eos_step_mask = eos_call["attention_mask"].eq(0.0)
    assert eos_step_mask.shape == (1, 1, 1, 7)
    assert torch.equal(eos_step_mask[0, 0, 0], extended_masks[0])
    assert torch.equal(eos_call["position_ids"], torch.tensor([[5]]))


def _make_sample_actions_model(*, predict_subtask, generate_at_inference):
    fake_language_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation=None))
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_inference_steps=1,
            chunk_size=2,
            max_action_dim=3,
            predict_subtask=predict_subtask,
            subtask_generate_at_inference=generate_at_inference,
        ),
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(model=SimpleNamespace(language_model=fake_language_model)),
            forward=lambda **kwargs: (None, object()),
        ),
        rtc_processor=None,
        subtask_seed_token_ids=torch.tensor([1, 2]),
        subtask_eos_token_id=5,
        generated=False,
        denoise_prefix_masks=None,
        _last_subtask_tokens=None,
    )

    def embed_prefix(images, img_masks, tokens, masks):
        bsize = tokens.shape[0]
        return (
            torch.zeros(bsize, 3, 4),
            torch.tensor([[True, False, True]], device=tokens.device).expand(bsize, 3),
            torch.zeros(bsize, 3, dtype=torch.bool, device=tokens.device),
        )

    def generate_subtask(**kwargs):
        model.generated = True
        extended_masks = torch.cat(
            [
                kwargs["prefix_pad_masks"],
                torch.ones(kwargs["prefix_pad_masks"].shape[0], 2, dtype=torch.bool),
            ],
            dim=1,
        )
        model._last_subtask_tokens = torch.tensor([[1, 2, 5]])
        return model._last_subtask_tokens, extended_masks, kwargs["past_key_values"]

    def denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        model.denoise_prefix_masks = prefix_pad_masks
        return torch.zeros_like(x_t)

    model.sample_noise = lambda shape, device: torch.ones(shape, device=device)
    model.embed_prefix = embed_prefix
    model._prepare_attention_masks_4d = lambda att_2d_masks: torch.where(
        att_2d_masks[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE
    )
    model._generate_subtask = generate_subtask
    model.denoise_step = denoise_step
    model._rtc_enabled = lambda: False
    model.sample_actions = PI05Pytorch.sample_actions.__get__(model)
    return model


def test_sample_actions_only_generates_subtask_when_both_inference_flags_are_enabled():
    tokens = torch.ones(1, 2, dtype=torch.long)
    masks = torch.ones(1, 2, dtype=torch.bool)

    disabled_model = _make_sample_actions_model(predict_subtask=False, generate_at_inference=True)
    disabled_model.sample_actions([], [], tokens, masks, noise=torch.zeros(1, 2, 3))
    assert not disabled_model.generated
    assert torch.equal(disabled_model.denoise_prefix_masks, torch.tensor([[True, False, True]]))

    skipped_model = _make_sample_actions_model(predict_subtask=True, generate_at_inference=False)
    skipped_model.sample_actions([], [], tokens, masks, noise=torch.zeros(1, 2, 3))
    assert not skipped_model.generated
    assert torch.equal(skipped_model.denoise_prefix_masks, torch.tensor([[True, False, True]]))

    enabled_model = _make_sample_actions_model(predict_subtask=True, generate_at_inference=True)
    enabled_model.sample_actions([], [], tokens, masks, noise=torch.zeros(1, 2, 3))
    assert enabled_model.generated
    assert torch.equal(
        enabled_model.denoise_prefix_masks,
        torch.tensor([[True, False, True, True, True]]),
    )


class _FakeTokenizer:
    def batch_decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens
        assert tokens == [[1, 2, 5]]
        return ["Subtask: pick; Progress: 0.5"]


def test_pi05_policy_predict_action_chunk_decodes_generated_subtask_text():
    class _FakeModel:
        def __init__(self):
            self._last_subtask_tokens = None

        def sample_actions(self, images, img_masks, tokens, masks, **kwargs):
            self._last_subtask_tokens = torch.tensor([[1, 2, 5]])
            return torch.zeros(1, 2, 3)

    policy = SimpleNamespace(
        config=SimpleNamespace(
            predict_subtask=True,
            output_features={ACTION: SimpleNamespace(shape=(2,))},
        ),
        model=_FakeModel(),
        _paligemma_tokenizer=_FakeTokenizer(),
        last_subtask_text="",
        _last_logged_subtask_text=None,
    )
    policy._preprocess_images = lambda batch: ([], [])
    policy.eval = lambda: None

    actions = PI05Policy.predict_action_chunk(
        policy,
        {
            OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
        },
    )

    assert actions.shape == (1, 2, 2)
    assert policy.last_subtask_text == "Subtask: pick; Progress: 0.5"
