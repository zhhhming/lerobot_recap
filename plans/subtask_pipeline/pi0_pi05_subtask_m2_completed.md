# PI0 / PI0.5 Subtask M2 Completion Notes

Date: 2026-07-03

## Scope

Completed milestone M2 from `pi0_pi05_subtask_generation_plan.md`:

- Added subtask AR config fields to PI0 and PI0.5 configs.
- Added config validation for `predict_subtask=True` with `train_expert_only=True`.
- Added a warning when inference decode length exceeds the training subtask token length.
- Added `SubtaskTextProcessorStep` to format raw `subtask` and `subtask_progress` into AR text.
- Extended `TokenizerProcessorStep` with explicit `tokenize_subtask` and `subtask_max_length` settings.
- Kept subtask tokenization disabled by default.
- Wired PI0 and PI0.5 preprocessors so subtask processing is enabled only when `config.predict_subtask=True`.
- Added PI0.5 prompt switch to omit the trailing `Action: ` when subtask prediction is enabled.

## Verification

Added `tests/processor/test_subtask_ar_processors.py`.

The test uses a local mock tokenizer and patches tokenizer loading in pipeline construction, so it does not require HuggingFace network access or a real annotated dataset.

Covered checks:

- default subtask behavior remains disabled;
- subtask text formatting handles batch, empty labels, missing labels, and missing progress;
- subtask tokenizer appends EOS and right pads to `subtask_max_length`;
- empty subtask text produces all-pad subtask tokens;
- tokenizer config and feature declarations include subtask fields when enabled;
- PI0 and PI0.5 pipelines include subtask steps only under `predict_subtask=True`;
- PI0.5 prompt removes `Action: ` only under subtask mode;
- config validation and warnings behave as intended.

## Remaining External Validation

This milestone does not cover model-side consumption of subtask tokens. The following remain for M3/M4:

- PI0.5 model forward path consumes subtask tokens and computes CE loss.
- Subtask dropout masks action attention to the subtask segment.
- Inference-time AR generation uses KV cache and exposes generated text.
