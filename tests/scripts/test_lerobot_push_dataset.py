#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import httpx

from lerobot.scripts import lerobot_push_dataset


def _write_cached_file(root: Path) -> Path:
    cached_file = lerobot_push_dataset._large_upload_cache_dir(root) / "videos" / "file.mp4.metadata"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_text("cached upload state")
    return cached_file


def test_prepare_large_upload_cache_marks_new_target(tmp_path: Path) -> None:
    lerobot_push_dataset._prepare_large_upload_cache(tmp_path, "owner/dataset", None)

    assert lerobot_push_dataset._large_upload_cache_marker(tmp_path).read_text() == "dataset:owner/dataset:main"


def test_prepare_large_upload_cache_preserves_matching_target(tmp_path: Path) -> None:
    cached_file = _write_cached_file(tmp_path)
    lerobot_push_dataset._mark_large_upload_cache_target(tmp_path, "owner/dataset", "dev")

    lerobot_push_dataset._prepare_large_upload_cache(tmp_path, "owner/dataset", "dev")

    assert cached_file.read_text() == "cached upload state"


def test_prepare_large_upload_cache_replaces_different_target(tmp_path: Path) -> None:
    cached_file = _write_cached_file(tmp_path)
    lerobot_push_dataset._mark_large_upload_cache_target(tmp_path, "owner/old-dataset", None)

    lerobot_push_dataset._prepare_large_upload_cache(tmp_path, "owner/new-dataset", None)

    assert not cached_file.exists()
    assert lerobot_push_dataset._large_upload_cache_marker(tmp_path).read_text() == "dataset:owner/new-dataset:main"


def test_prepare_large_upload_cache_replaces_unmarked_legacy_cache(tmp_path: Path) -> None:
    cached_file = _write_cached_file(tmp_path)

    lerobot_push_dataset._prepare_large_upload_cache(tmp_path, "owner/dataset", None)

    assert not cached_file.exists()
    assert lerobot_push_dataset._large_upload_cache_marker(tmp_path).read_text() == "dataset:owner/dataset:main"


def test_tls_client_uses_finite_timeouts(monkeypatch) -> None:
    captured_factory = None

    def capture_client_factory(factory) -> None:
        nonlocal captured_factory
        captured_factory = factory

    monkeypatch.setattr(lerobot_push_dataset, "set_client_factory", capture_client_factory)

    lerobot_push_dataset._configure_tls(max_tls_1_2=True)

    assert captured_factory is not None
    with captured_factory() as client:
        assert isinstance(client, httpx.Client)
        assert client.timeout.connect == lerobot_push_dataset.HUB_CONNECT_TIMEOUT_SECONDS
        assert client.timeout.read == lerobot_push_dataset.HUB_IO_TIMEOUT_SECONDS
        assert client.timeout.write == lerobot_push_dataset.HUB_IO_TIMEOUT_SECONDS


def test_configure_xet_disables_xet_after_import(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.setattr(lerobot_push_dataset.hf_constants, "HF_HUB_DISABLE_XET", False)

    lerobot_push_dataset._configure_xet(disable_xet=True)

    assert lerobot_push_dataset.os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert lerobot_push_dataset.hf_constants.HF_HUB_DISABLE_XET is True
