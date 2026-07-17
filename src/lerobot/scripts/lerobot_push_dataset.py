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

"""
Upload a locally recorded LeRobot dataset to the Hugging Face Hub.

Examples:

```shell
lerobot-push-dataset --repo_id ming326/nero_try

lerobot-push-dataset \
    --repo_id ming326/nero_try \
    --dry-run

lerobot-push-dataset \
    --repo_id ming326/nero_try \
    --proxy http://127.0.0.1:1080
```
"""

import argparse
import importlib.util
import logging
import os
import shutil
import ssl
from pathlib import Path

import httpx
from huggingface_hub import constants as hf_constants
from huggingface_hub import set_client_factory
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.utils._http import hf_request_event_hook

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


logger = logging.getLogger(__name__)
LARGE_UPLOAD_CACHE_MARKER = "lerobot_push_dataset_upload_target.txt"
HUB_CONNECT_TIMEOUT_SECONDS = 10.0
HUB_IO_TIMEOUT_SECONDS = 60.0


def _resolve_dataset_root(repo_id: str, root: str | Path | None) -> Path:
    return Path(root).expanduser() if root is not None else HF_LEROBOT_HOME / repo_id


def _parse_tags(tags: str | None) -> list[str] | None:
    if tags is None:
        return None
    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return parsed_tags or None


def _normalize_proxy(proxy: str) -> str:
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    if proxy.startswith("socks") and importlib.util.find_spec("socksio") is None:
        raise RuntimeError(
            "SOCKS proxy support requires the 'socksio' package in this environment. "
            "Use an HTTP proxy URL such as http://127.0.0.1:1080, or install socksio before using socks5://."
        )
    return proxy


def _configure_proxy(proxy: str | None) -> None:
    if proxy is None:
        return
    normalized_proxy = _normalize_proxy(proxy)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[key] = normalized_proxy
    logger.info("Using proxy %s", normalized_proxy)


def _configure_xet(disable_xet: bool) -> None:
    if not disable_xet:
        return

    # huggingface_hub reads this environment variable into a module constant at
    # import time. Update both so the CLI flag also works after LeRobot imports.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    hf_constants.HF_HUB_DISABLE_XET = True
    logger.info("Disabled Hugging Face Xet transfers; using standard HTTP/LFS uploads")


def _configure_tls(max_tls_1_2: bool) -> None:
    if not max_tls_1_2:
        return

    ssl_context = ssl.create_default_context()
    ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2

    def client_factory() -> httpx.Client:
        return httpx.Client(
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=httpx.Timeout(
                HUB_IO_TIMEOUT_SECONDS,
                connect=HUB_CONNECT_TIMEOUT_SECONDS,
            ),
            verify=ssl_context,
        )

    set_client_factory(client_factory)
    logger.info("Configured Hugging Face Hub HTTP client with maximum TLS version 1.2")


def _check_local_dataset(root: Path) -> None:
    info_path = root / INFO_PATH
    if not info_path.exists():
        raise FileNotFoundError(
            f"Could not find a local LeRobot dataset at '{root}'. "
            f"Expected metadata file '{info_path}'. "
            "Pass --root if the dataset was recorded somewhere else."
        )


def _large_upload_cache_dir(root: Path) -> Path:
    return root / ".cache" / "huggingface" / "upload"


def _large_upload_cache_marker(root: Path) -> Path:
    return root / ".cache" / "huggingface" / LARGE_UPLOAD_CACHE_MARKER


def _large_upload_target(repo_id: str, branch: str | None) -> str:
    return f"dataset:{repo_id}:{branch or 'main'}"


def _prepare_large_upload_cache(root: Path, repo_id: str, branch: str | None) -> None:
    """Prepare resumable upload state for exactly one Hub repo target.

    huggingface_hub stores upload state under the local dataset directory and
    documents that the same folder must not be uploaded to multiple repos unless
    this cache is removed first. Reusing it can mark files as already committed
    and produce an incomplete target repository. The target marker must be
    written before uploading so an interrupted upload can resume on the next run.
    """
    upload_cache = _large_upload_cache_dir(root)
    target = _large_upload_target(repo_id, branch)
    marker = _large_upload_cache_marker(root)
    if upload_cache.exists():
        previous_target = marker.read_text().strip() if marker.exists() else None
        if previous_target != target:
            logger.warning(
                "Removing stale Hugging Face large-upload cache at '%s' before uploading to %s. "
                "This prevents reusing upload state from a different dataset repo.",
                upload_cache,
                target,
            )
            shutil.rmtree(upload_cache)

    _mark_large_upload_cache_target(root, repo_id, branch)


def _mark_large_upload_cache_target(root: Path, repo_id: str, branch: str | None) -> None:
    marker = _large_upload_cache_marker(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(_large_upload_target(repo_id, branch))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True, help="Hugging Face dataset repo id, e.g. ming326/nero_try.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Exact local dataset folder. Defaults to $HF_LEROBOT_HOME/{repo_id}.",
    )
    parser.add_argument("--branch", default=None, help="Optional Hub branch to upload to.")
    parser.add_argument("--private", action="store_true", help="Create or keep the Hub dataset repo private.")
    parser.add_argument("--tags", default=None, help="Comma-separated dataset tags.")
    parser.add_argument("--license", default="apache-2.0", help="Dataset card license. Defaults to apache-2.0.")
    parser.add_argument("--proxy", default=None, help="Proxy URL, e.g. http://127.0.0.1:1080.")
    parser.add_argument(
        "--disable-xet",
        action="store_true",
        help="Disable Hugging Face Xet transfers and use standard HTTP/LFS uploads.",
    )
    parser.add_argument(
        "--tls-max-1-2",
        action="store_true",
        help="Limit HTTPS connections to TLS 1.2 for proxy/node compatibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize the local dataset without uploading.",
    )
    parser.add_argument(
        "--no-push-videos",
        action="store_true",
        help="Upload metadata and parquet files but skip the videos/ directory.",
    )
    parser.add_argument("--no-tag-version", action="store_true", help="Do not create/update the code version tag.")
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use Hugging Face Hub upload_large_folder for very large datasets.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for --upload-large-folder. Try 1 or 2 on unstable proxies.",
    )
    parser.add_argument(
        "--allow-patterns",
        default=None,
        help="Optional glob pattern(s) passed to huggingface_hub upload_folder. "
        "Use comma-separated values for multiple patterns.",
    )
    return parser


def _parse_allow_patterns(allow_patterns: str | None) -> list[str] | str | None:
    if allow_patterns is None:
        return None
    patterns = [pattern.strip() for pattern in allow_patterns.split(",") if pattern.strip()]
    if not patterns:
        return None
    return patterns[0] if len(patterns) == 1 else patterns


def main() -> None:
    init_logging()
    args = build_parser().parse_args()

    _configure_proxy(args.proxy)
    _configure_xet(args.disable_xet)
    _configure_tls(args.tls_max_1_2)

    root = _resolve_dataset_root(args.repo_id, args.root)
    _check_local_dataset(root)

    dataset = LeRobotDataset(args.repo_id, root=root)
    if dataset.num_episodes == 0:
        raise RuntimeError(f"Dataset '{args.repo_id}' at '{root}' has no episodes to upload.")

    logger.info(
        "Loaded dataset '%s' from '%s' (%s episodes, %s frames)",
        args.repo_id,
        root,
        dataset.num_episodes,
        dataset.num_frames,
    )
    if args.dry_run:
        logger.info("Dry run complete; no upload was performed.")
        return

    logger.info("Uploading dataset '%s' to the Hugging Face Hub", args.repo_id)
    if args.num_workers is not None and not args.upload_large_folder:
        logger.warning("--num-workers is ignored unless --upload-large-folder is set.")
    if args.upload_large_folder:
        _prepare_large_upload_cache(root, args.repo_id, args.branch)
    try:
        dataset.push_to_hub(
            branch=args.branch,
            tags=_parse_tags(args.tags),
            license=args.license,
            tag_version=not args.no_tag_version,
            push_videos=not args.no_push_videos,
            private=args.private,
            allow_patterns=_parse_allow_patterns(args.allow_patterns),
            upload_large_folder=args.upload_large_folder,
            upload_large_folder_num_workers=args.num_workers,
        )
    except HfHubHTTPError:
        logger.exception("Hub upload failed. Check your Hugging Face token, repo permissions, and network/proxy.")
        raise
    logger.info("Uploaded dataset to https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
