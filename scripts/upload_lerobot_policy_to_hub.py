#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_PATH = Path(
    "/datastore01/hongming/lerobot_outputs/"
    "pi0_nero_candle_relative_bs256_20260515_145730/"
    "checkpoints/003000/pretrained_model"
)

REQUIRED_FILES = ("config.json", "model.safetensors")
DEFAULT_ALLOW_PATTERNS = ("*.json", "*.safetensors", "*.md", "*.yaml", "*.yml")
DEFAULT_IGNORE_PATTERNS = ("*.tmp", "*.log", "__pycache__/*", ".DS_Store")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a LeRobot pretrained_model checkpoint folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to the LeRobot pretrained_model directory. Default: {DEFAULT_CHECKPOINT_PATH}",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Destination Hub model repo, for example: ming326/pi0_nero_candle_relative_bs256",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    parser.add_argument("--revision", default="main", help="Branch or revision to upload to. Default: main.")
    parser.add_argument("--create-pr", action="store_true", help="Create a Hub pull request instead of pushing directly.")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to the cached login token.")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Ignore proxy and custom Hugging Face endpoint environment variables before uploading.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload LeRobot policy checkpoint",
        help="Commit message for the checkpoint upload.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        help="Allowed upload pattern. Can be passed multiple times. Defaults to json/safetensors/md/yaml.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        help="Ignored upload pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true",
        help="Do not generate a README.md model card when the checkpoint folder does not contain one.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded and exit.")
    return parser.parse_args()


def resolve_checkpoint_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name != "pretrained_model" and (path / "pretrained_model").is_dir():
        return path / "pretrained_model"
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return data


def validate_checkpoint(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{path} is missing required LeRobot policy file(s): {', '.join(missing)}"
        )


def iter_upload_files(path: Path, allow_patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file_path.relative_to(path)
        if any(rel.match(pattern) for pattern in allow_patterns):
            files.append(rel)
    return files


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def generate_model_card(repo_id: str, checkpoint_path: Path) -> str:
    config = load_json(checkpoint_path / "config.json")
    train_config = load_json(checkpoint_path / "train_config.json")

    policy_type = config.get("type") or train_config.get("policy", {}).get("type") or "lerobot-policy"
    dataset_repo_id = train_config.get("dataset", {}).get("repo_id")
    license_name = config.get("license") or "apache-2.0"
    tags = sorted({"lerobot", "robotics", str(policy_type)})

    metadata_lines = [
        "---",
        f"license: {license_name}",
        "library_name: lerobot",
        "pipeline_tag: robotics",
        "tags:",
        *(f"- {tag}" for tag in tags),
    ]
    if dataset_repo_id:
        metadata_lines.extend(["datasets:", f"- {dataset_repo_id}"])
    metadata_lines.append("---")

    details = [
        f"# {repo_id.split('/')[-1]}",
        "",
        "This repository contains a LeRobot policy checkpoint uploaded from a local training run.",
        "",
        "## Files",
        "",
        "- `config.json`: policy configuration",
        "- `model.safetensors`: policy weights",
        "- `train_config.json`: training configuration, when available",
        "- `policy_preprocessor*.json` / `policy_postprocessor*.json`: preprocessing pipelines",
        "- `*.safetensors`: normalization and processor state files",
        "",
        "## Load",
        "",
        "```python",
        "from lerobot.configs.policies import PreTrainedConfig",
        "from lerobot.policies.factory import get_policy_class",
        "",
        f'repo_id = "{repo_id}"',
        "config = PreTrainedConfig.from_pretrained(repo_id)",
        "policy_cls = get_policy_class(config.type)",
        "policy = policy_cls.from_pretrained(repo_id, config=config)",
        "```",
    ]

    if dataset_repo_id:
        details.extend(["", f"Training dataset: `{dataset_repo_id}`"])

    return "\n".join(metadata_lines + [""] + details) + "\n"


def print_dry_run(checkpoint_path: Path, repo_id: str, allow_patterns: list[str]) -> None:
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Repo ID:    {repo_id}")
    print("Files:")
    for rel_path in iter_upload_files(checkpoint_path, allow_patterns):
        size = (checkpoint_path / rel_path).stat().st_size
        print(f"  {rel_path} ({format_size(size)})")


def clear_hf_network_env() -> None:
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HF_ENDPOINT",
        "HF_INFERENCE_ENDPOINT",
    )
    removed = [key for key in keys if key in os.environ]
    for key in removed:
        os.environ.pop(key, None)
    if removed:
        print(f"Ignoring network environment variables: {', '.join(removed)}")


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    allow_patterns = args.allow_patterns or list(DEFAULT_ALLOW_PATTERNS)
    ignore_patterns = args.ignore_patterns or list(DEFAULT_IGNORE_PATTERNS)

    validate_checkpoint(checkpoint_path)

    if args.dry_run:
        print_dry_run(checkpoint_path, args.repo_id, allow_patterns)
        if not args.no_model_card and not (checkpoint_path / "README.md").exists():
            print("README.md will be generated and uploaded in a second commit.")
        return 0

    if args.direct:
        clear_hf_network_env()

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    repo = api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading {checkpoint_path} to https://huggingface.co/{repo.repo_id}")
    commit_info = api.upload_folder(
        repo_id=repo.repo_id,
        repo_type="model",
        folder_path=checkpoint_path,
        revision=args.revision,
        create_pr=args.create_pr,
        commit_message=args.commit_message,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    readme_path = checkpoint_path / "README.md"
    if not args.no_model_card and not readme_path.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as f:
            f.write(generate_model_card(repo.repo_id, checkpoint_path))
            temp_readme = Path(f.name)
        try:
            api.upload_file(
                repo_id=repo.repo_id,
                repo_type="model",
                path_or_fileobj=temp_readme,
                path_in_repo="README.md",
                revision=args.revision,
                create_pr=args.create_pr,
                commit_message="Add LeRobot model card",
            )
        finally:
            temp_readme.unlink(missing_ok=True)

    commit_url = getattr(commit_info, "commit_url", None) or str(commit_info)
    print(f"Done: {commit_url}")
    print(f"Model repo: https://huggingface.co/{repo.repo_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            print(
                "Error: huggingface_hub is not installed in this Python environment.",
                file=sys.stderr,
            )
            print(
                "Install it with: python -m pip install 'huggingface-hub>=1.0,<2.0'",
                file=sys.stderr,
            )
            raise SystemExit(1)
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Tip: run `huggingface-cli login` first, or pass `--token hf_...`.",
            file=sys.stderr,
        )
        raise SystemExit(1)
