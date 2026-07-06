# PI0 / PI0.5 Subtask M3 Completion Notes

Date: 2026-07-03

## Scope

Completed milestone M3 from `pi0_pi05_subtask_generation_plan.md`:

- `PI05Pytorch.embed_prefix()` accepts optional `subtask_tokens` and `subtask_masks`.
- Subtask token embeddings are appended after image/prompt prefix tokens.
- Subtask attention markers use `[1] * S`, reusing existing `make_att_2d_masks()` causal block behavior.
- `PI05Pytorch.forward()` accepts optional subtask tensors and returns `(fm_losses, ce_loss_per_sample)`.
- Teacher-forced subtask CE is computed from prefix outputs over the subtask segment.
- CE is a per-sample tensor and all-pad subtask rows produce exactly `0`.
- Subtask dropout masks suffix/action rows from attending to subtask columns during training.
- `PI05Policy.forward()` passes subtask tensors when `config.predict_subtask=True`.
- `PI05Policy.forward()` aggregates `fm_loss + subtask_ce_loss_weight * ce_loss` for both `mean` and `none` reductions.
- Loss logging now includes `fm_loss` and `ce_loss` in addition to existing entries.

## Verification

Added `tests/policies/pi0_pi05/test_pi05_subtask_training.py`.

The test is intentionally local and lightweight:

- no GPU required;
- no HuggingFace weights required;
- no real annotated dataset required;
- no full PI0.5 model instantiation required.

Covered checks:

- `make_att_2d_masks()` produces the expected prefix/subtask/action layout;
- subtask CE uses shifted next-token targets and ignores padded tokens;
- all-empty subtask rows produce `0` CE;
- subtask dropout only masks suffix/action rows to subtask columns;
- policy `mean` reduction combines FM and CE losses correctly;
- policy `none` reduction combines per-sample FM and CE losses correctly;
- missing subtask tensors preserve FM-only behavior.

## Optional GPU Smoke

If a local 4090/CUDA environment has cached PaliGemma weights, an additional manual smoke is useful after M3:

1. Instantiate a small PI0.5 config with `predict_subtask=True`, `paligemma_variant="gemma_300m"`, and `action_expert_variant="gemma_300m"`.
2. Build a one-sample fake batch containing image, state, action, language tokens, and subtask tokens.
3. Run `policy.forward(batch)` and verify finite `loss`, `fm_loss`, and `ce_loss`.
4. Run `loss.backward()` and verify no dtype/mask/runtime error.

This is not part of the default M3 gate because it depends on CUDA visibility, cached model weights, and available test dependencies.

## Remaining Work

M4 still needs inference-time subtask AR generation:

- `_generate_subtask()` with KV-cache decoding;
- `sample_actions()` integration;
- generated text exposure/logging from `PI05Policy`.
