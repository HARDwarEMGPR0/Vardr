"""Run the market intelligence pipeline and optionally send the prompt to Claude."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_trade_ideas import build_prompt, load_snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vardr market intelligence through Claude")
    parser.add_argument("--vardr-leader-api-url", type=str, default=None)
    parser.add_argument("--leader-markets-json", type=str, default=None)
    parser.add_argument("--history-jsonl", type=str, default="data/history.jsonl")
    parser.add_argument("--snapshot-out", type=str, default="data/latest_snapshot.json")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5")
    parser.add_argument("--no-claude", action="store_true")
    return parser.parse_args(argv)


def _fetch_snapshot_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "fetch_snapshots.py"),
        "--history-jsonl",
        args.history_jsonl,
    ]
    if args.vardr_leader_api_url:
        command.extend(["--vardr-leader-api-url", args.vardr_leader_api_url])
    elif args.leader_markets_json:
        command.extend(["--leader-markets-json", args.leader_markets_json])
    return command


def run_fetch_snapshot(args: argparse.Namespace) -> int:
    result = subprocess.run(
        _fetch_snapshot_command(args),
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    if result.stderr:
        sys.stderr.write(result.stderr)

    if result.returncode != 0:
        sys.stderr.write(f"fetch_snapshots.py failed with exit code {result.returncode}\n")
        return result.returncode

    try:
        json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"fetch_snapshots.py did not return valid JSON: {exc}\n")
        return 1

    snapshot_path = Path(args.snapshot_out)
    if not snapshot_path.is_absolute():
        snapshot_path = ROOT_DIR / snapshot_path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(result.stdout, encoding="utf-8")
    return 0


def call_claude(prompt: str, model: str) -> str:
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError as exc:
        raise RuntimeError("Anthropic Python SDK missing. Run: pip install anthropic") from exc

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fetch_code = run_fetch_snapshot(args)
    if fetch_code != 0:
        return fetch_code

    snapshot = load_snapshot(args.snapshot_out)
    prompt = build_prompt(snapshot)

    if args.no_claude:
        print(prompt)
        return 0

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.stderr.write("Error: ANTHROPIC_API_KEY is required unless --no-claude is passed.\n")
        return 2

    try:
        memo = call_claude(prompt, args.model)
    except RuntimeError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    print(memo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
