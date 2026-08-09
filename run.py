"""Command-line entry point for the support AI suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parse_ticket(args: argparse.Namespace) -> dict[str, Any]:
    """Load one ticket object from an inline JSON value or a JSON file."""
    if args.ticket_json is not None:
        source = "--ticket-json"
        raw_ticket = args.ticket_json
    else:
        source = str(args.ticket_file)
        raw_ticket = args.ticket_file.read_text(encoding="utf-8")

    try:
        ticket = json.loads(raw_ticket)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} must contain a valid JSON object: {exc.msg}") from exc

    if not isinstance(ticket, dict):
        raise ValueError(f"{source} must contain one ticket JSON object, not {type(ticket).__name__}.")
    return ticket


def create_app():
    """Create the HTTP app with the routes available for the current delivery track."""
    from fastapi import FastAPI

    from src.account_brief.api import router as account_brief_router
    from src.triage.api import router as triage_router

    app = FastAPI(title="Support AI Suite")
    app.include_router(triage_router)
    app.include_router(account_brief_router)
    return app


def _run_triage(args: argparse.Namespace) -> int:
    from src.triage.agent import classify_ticket

    ticket = _parse_ticket(args)
    print("Draft response (streaming):")

    def write_token(token: str) -> None:
        print(token, end="", flush=True)

    result = classify_ticket(ticket, on_draft_token=write_token)
    print("\n\nValidated triage result:")
    print(result.model_dump_json(indent=2))
    return 0


def _run_brief(args: argparse.Namespace) -> int:
    from src.account_brief.agent import AccountNotFoundError, generate_brief

    try:
        brief = generate_brief(args.account_id)
    except AccountNotFoundError:
        print(f"error: account_id {args.account_id!r} was not found in data/accounts.json.")
        return 2

    print(brief.model_dump_json(indent=2))
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def _run_eval(_: argparse.Namespace) -> int:
    from src.eval.run_eval import REPORT_JSON_PATH, REPORT_MD_PATH
    from src.eval.run_eval import main as run_eval_main

    results = run_eval_main()

    # hard_gate_pass is tri-state: True (passed) / False (genuinely failed) / None
    # (SKIPPED — no provider key in this environment). Skips are neither passes nor
    # failures for exit-code purposes: CI has no keys by design (no secrets in the
    # workflow) and must still exit 0 when nothing that actually ran, failed.
    total = len(results)
    n_pass = sum(1 for r in results if r.hard_gate_pass is True)
    n_fail = sum(1 for r in results if r.hard_gate_pass is False)
    n_skip = sum(1 for r in results if r.hard_gate_pass is None)
    scored = [r.quality_score for r in results if r.quality_score is not None]
    avg_quality = sum(scored) / len(scored) if scored else None

    print(f"Eval: {n_pass}/{total} cases passed hard gates, {n_fail} failed, {n_skip} skipped.")
    if avg_quality is not None:
        print(f"Average quality_score (judged cases only, n={len(scored)}): {avg_quality:.3f}")
    else:
        print("Average quality_score: n/a (no cases were judged)")

    for r in results:
        if r.hard_gate_pass is False:
            print(f"  FAIL {r.case_id}: {r.notes}")
        elif r.hard_gate_pass is None:
            print(f"  SKIP {r.case_id}: {r.notes}")

    for path in (REPORT_JSON_PATH, REPORT_MD_PATH):
        print(f"{'wrote' if path.is_file() else 'MISSING'}: {path}")

    return 0 if n_fail == 0 else 1


def _run_ui(args: argparse.Namespace) -> int:
    app_path = Path(__file__).resolve().parent / "src" / "ui" / "app.py"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Support AI Suite command-line interface")
    subcommands = parser.add_subparsers(dest="command", required=True)

    triage = subcommands.add_parser("triage", help="Classify one ticket JSON object")
    ticket_source = triage.add_mutually_exclusive_group(required=True)
    ticket_source.add_argument("--ticket-file", type=Path, help="Path to one ticket JSON object")
    ticket_source.add_argument("--ticket-json", help="Inline ticket JSON object")
    triage.set_defaults(handler=_run_triage)

    brief = subcommands.add_parser("brief", help="Generate an account brief")
    brief.add_argument("--account-id", required=True, help="Account ID from data/accounts.json")
    brief.set_defaults(handler=_run_brief)

    serve = subcommands.add_parser("serve", help="Start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1", help="Interface to bind")
    serve.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve.set_defaults(handler=_run_serve)

    eval_parser = subcommands.add_parser("eval", help="Run the Task 3 eval harness (writes eval_report.json/.md)")
    eval_parser.set_defaults(handler=_run_eval)

    ui = subcommands.add_parser("ui", help="Start the Streamlit UI")
    ui.add_argument("--host", default="127.0.0.1", help="Interface to bind")
    ui.add_argument("--port", type=int, default=8501, help="Port to bind")
    ui.set_defaults(handler=_run_ui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except Exception as exc:
        # Broad on purpose: triage/brief call live LLM providers directly (unlike
        # `eval`, which already catches per-case failures in scorer.py, and the
        # FastAPI routes, which already wrap agent calls in HTTPException). Found via
        # fresh-clone testing with an empty .env: an empty GROQ_API_KEY produces
        # groq.APIConnectionError (masking the real "Illegal header value" auth
        # cause several layers down) — narrower except clauses let that reach the
        # user as a raw traceback instead of a clean one-line error.
        print(f"error: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
