# PI0 / PI0.5 Subtask M1 Completion Notes

Date: 2026-07-03

## Scope

Completed milestone M1 from `pi0_pi05_subtask_generation_plan.md`:

- `src/lerobot/scripts/lerobot_annotate_subtask.py`
  - `export_extras()` now writes a `subtask_progress` `float32` column next to `subtask`.
  - Progress is computed inside contiguous non-empty subtask segments.
  - Empty or unannotated frames get progress `0.0`.
  - Existing `extras.parquet` files keep unrelated columns, while old `subtask` and `subtask_progress` columns are replaced.
- `src/lerobot/scripts/lerobot_build_dataset.py`
  - extras schema loading now maps pyarrow floating scalar columns explicitly to valid LeRobot feature dtypes (`float16`, `float32`, `float64`).
- `src/lerobot/processor/converters.py`
  - `subtask_progress` is routed into `complementary_data` with `subtask`.

## Verification

Added `tests/scripts/test_subtask_progress_data_pipeline.py`.

The test constructs minimal raw-run directories and `extras.parquet` files under pytest `tmp_path`, so it does not require a real annotated dataset.

Covered checks:

- `extras.parquet` contains both `subtask` and `subtask_progress`.
- Progress for contiguous labels is linear and ends at `1.0`.
- Empty and missing labels produce progress `0.0`.
- Existing unrelated extras columns are preserved.
- Stale `subtask` / `subtask_progress` columns are replaced during export.
- `_load_extras_schema()` maps `subtask_progress` to `float32`.
- `batch_to_transition()` routes `subtask_progress` into complementary data.

## Remaining External Validation

Because there is currently no real local dataset containing subtask annotations, the full raw-run-to-LeRobotDataset integration should still be checked later with an actual annotated raw run:

1. Annotate and export `extras.parquet`.
2. Run `lerobot-build-dataset`.
3. Confirm `dataset[i]["subtask"]` is a string and `dataset[i]["subtask_progress"]` is a float value.
