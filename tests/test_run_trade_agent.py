from __future__ import annotations

import json
import subprocess
from pathlib import Path

import scripts.run_trade_agent as runner


def _snapshot_json() -> str:
    return json.dumps(
        {
            "run_id": "test-run",
            "intelligence": {
                "lag_signals": [],
                "review_candidates": [],
                "opportunities": [],
                "reference_event_clusters": [],
            },
            "scores": {"polymarket": []},
            "polymarket_snapshots": [],
        }
    )


def test_missing_leader_source_exits_clearly(capsys) -> None:
    code = runner.main(["--no-claude"])

    assert code != 0
    assert "provide --vardr-leader-api-url or --leader-markets-json" in capsys.readouterr().err


def test_no_claude_prints_prompt(monkeypatch, tmp_path, capsys) -> None:
    snapshot_out = tmp_path / "latest_snapshot.json"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=_snapshot_json(), stderr="debug\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "build_prompt", lambda snapshot: f"PROMPT {snapshot['run_id']}")

    code = runner.main([
        "--leader-markets-json",
        "leaders.json",
        "--snapshot-out",
        str(snapshot_out),
        "--no-claude",
    ])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "PROMPT test-run"
    assert "debug" in captured.err


def test_missing_anthropic_api_key_errors_clearly(monkeypatch, tmp_path, capsys) -> None:
    snapshot_out = tmp_path / "latest_snapshot.json"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=_snapshot_json(), stderr="")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "build_prompt", lambda snapshot: "PROMPT")

    code = runner.main(["--leader-markets-json", "leaders.json", "--snapshot-out", str(snapshot_out)])

    assert code != 0
    assert "ANTHROPIC_API_KEY is required" in capsys.readouterr().err


def test_snapshot_file_is_written(monkeypatch, tmp_path) -> None:
    snapshot_out = tmp_path / "latest_snapshot.json"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=_snapshot_json(), stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "build_prompt", lambda snapshot: "PROMPT")

    code = runner.main([
        "--leader-markets-json",
        "leaders.json",
        "--snapshot-out",
        str(snapshot_out),
        "--no-claude",
    ])

    assert code == 0
    assert json.loads(snapshot_out.read_text(encoding="utf-8"))["run_id"] == "test-run"


def test_build_prompt_is_reused(monkeypatch, tmp_path) -> None:
    snapshot_out = tmp_path / "latest_snapshot.json"
    calls = []

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=_snapshot_json(), stderr="")

    def fake_build_prompt(snapshot):
        calls.append(snapshot)
        return "PROMPT"

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "build_prompt", fake_build_prompt)

    code = runner.main([
        "--vardr-leader-api-url",
        "http://localhost:8000/leader-markets",
        "--snapshot-out",
        str(snapshot_out),
        "--no-claude",
    ])

    assert code == 0
    assert len(calls) == 1
    assert calls[0]["run_id"] == "test-run"


def test_subprocess_failure_returns_nonzero(monkeypatch, tmp_path, capsys) -> None:
    snapshot_out = tmp_path / "latest_snapshot.json"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=7, stdout="", stderr="fetch failed\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    code = runner.main([
        "--leader-markets-json",
        "leaders.json",
        "--snapshot-out",
        str(snapshot_out),
        "--no-claude",
    ])

    captured = capsys.readouterr()
    assert code == 7
    assert "fetch failed" in captured.err
    assert "fetch_snapshots.py failed with exit code 7" in captured.err
    assert not snapshot_out.exists()


def test_call_claude_output_is_printed(monkeypatch, tmp_path, capsys) -> None:
    snapshot_out = tmp_path / "latest_snapshot.json"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=_snapshot_json(), stderr="")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "build_prompt", lambda snapshot: "PROMPT")
    monkeypatch.setattr(runner, "call_claude", lambda prompt, model: f"MEMO {model} {prompt}")

    code = runner.main([
        "--leader-markets-json",
        "leaders.json",
        "--snapshot-out",
        str(snapshot_out),
        "--model",
        "claude-test",
    ])

    assert code == 0
    assert capsys.readouterr().out.strip() == "MEMO claude-test PROMPT"
