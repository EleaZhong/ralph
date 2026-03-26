#!/usr/bin/env python3
"""
Ralph Wiggum Method as in https://sidbharath.com/blog/ralph-wiggum-claude-code/
run Claude Code in a loop until completion.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import yaml

# ---------------------------------------------------------------------------
# Modular prompt parts registry
# ---------------------------------------------------------------------------

@dataclass
class PromptPart:
    name: str           # Used in --parts list
    default: bool       # Enabled by default?
    content: str        # Prompt text (may contain {format} vars)


COMPLETION_SIGNAL = "COMPLETE"
PROGRESS_FILE = "PROGRESS.md"

# Default --allowedTools for non-interactive mode: wide but safe.
# Read/Grep/Glob are auto-approved by Claude Code so we don't need to list them.
DEFAULT_ALLOWED_TOOLS = [
    "Edit",
    "Write",
    "Bash",
    "Agent",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
]

# Default parts directory: ./parts/ next to this script
DEFAULT_PARTS_DIR = pathlib.Path(__file__).parent / "parts"


def load_parts(parts_dir: pathlib.Path) -> list[PromptPart]:
    """Load all PromptPart definitions from YAML files in a directory."""
    parts = []
    if not parts_dir.is_dir():
        return parts
    for yaml_file in sorted(parts_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        parts.append(PromptPart(
            name=data["name"],
            default=data.get("default", False),
            content=data.get("content", "").strip(),
        ))
    return parts

# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompt(
    user_prompt: str,
    active_parts: list[PromptPart],
) -> str:
    fmt = dict(
        completion_signal=COMPLETION_SIGNAL,
        progress_file=PROGRESS_FILE,
    )

    sections = [user_prompt.strip()]

    lines = ["", "## Iteration Protocol"]
    for part in active_parts:
        lines.append(f"- {part.content.format(**fmt)}")
    lines.append(
        f'- When ALL completion criteria are met, output the exact line: {COMPLETION_SIGNAL}'
    )
    sections.append("\n".join(lines))

    return "\n".join(sections)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parts_help(parts_dir: pathlib.Path) -> str:
    """Build help text listing all available parts with their prompts."""
    parts = load_parts(parts_dir)
    if not parts:
        return "No parts found."
    lines = []
    for p in parts:
        default_tag = " (default)" if p.default else ""
        lines.append(f"  {p.name}{default_tag}:")
        # Wrap content to ~70 chars indented
        for content_line in p.content.splitlines():
            lines.append(f"    {content_line.strip()}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parts_help = _parts_help(DEFAULT_PARTS_DIR)

    p = argparse.ArgumentParser(
        description="Run Claude Code in a loop until completion.",
        epilog=(
            "Anything after -- is passed directly to `claude`.\n\n"
            "Available parts:\n" + parts_help
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "prompt_file",
        nargs="?",
        default=None,
        help="Path to a prompt file (.md, .txt). Omit to type inline.",
    )
    p.add_argument(
        "-p", "--prompt",
        default=None,
        help="Inline prompt string (used if no file given).",
    )
    p.add_argument(
        "-n", "--max-iterations",
        type=int,
        default=10,
        help="Maximum loop iterations (default: 10).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assembled prompt and exit.",
    )
    p.add_argument(
        "--parts",
        nargs="*",
        default=None,
        help=(
            "List of prompt parts to enable (by name). "
            "If omitted, all parts with default=true are enabled. "
            "Example: --parts small-changes tests progress"
        ),
    )
    p.add_argument(
        "--parts-dir",
        default=None,
        help=f"Directory containing part YAML files (default: {DEFAULT_PARTS_DIR}).",
    )
    p.add_argument(
        "--safe",
        action="store_true",
        help=(
            "Use --allowedTools instead of --dangerously-skip-permissions. "
            "By default, runs in yolo mode (skip permissions + IS_SANDBOX=1)."
        ),
    )

    return p


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        with open(args.prompt_file) as f:
            return f.read()
    if args.prompt:
        return args.prompt
    # Interactive: read from stdin
    print("Enter your prompt (Ctrl-D to finish):", file=sys.stderr)
    return sys.stdin.read()


def resolve_active_parts(
    all_parts: list[PromptPart], selected: list[str] | None,
) -> list[PromptPart]:
    """Return the list of active PromptParts based on --parts selection.

    If selected is None (flag omitted), use each part's default.
    If selected is an explicit list, enable exactly those parts.
    """
    if selected is None:
        return [p for p in all_parts if p.default]
    by_name = {p.name: p for p in all_parts}
    active = []
    for name in selected:
        if name not in by_name:
            available = ", ".join(sorted(by_name))
            print(
                f"Error: unknown part '{name}'. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        active.append(by_name[name])
    return active

# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------

def _run_streaming(
    cmd: list[str], prompt: str, env: dict[str, str],
) -> tuple[str, int]:
    """Run claude -p with stream-json, printing text as it arrives.

    Returns (collected_assistant_text, returncode).
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin is not None
    proc.stdin.write(prompt)
    proc.stdin.close()

    collected: list[str] = []
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            raw_line = raw_line.rstrip("\n")
            if not raw_line:
                continue
            try:
                msg = json.loads(raw_line)
            except json.JSONDecodeError:
                print(raw_line, flush=True)
                collected.append(raw_line + "\n")
                continue

            # stream-json schema:
            #   assistant: {type: "assistant", message: {content: [{type: "text", text: "..."}]}}
            #   result:    {type: "result", result: "final text"}
            msg_type = msg.get("type", "")

            if msg_type == "assistant":
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            print(text, end="", flush=True)
                            collected.append(text)
            elif msg_type == "result":
                result_text = msg.get("result", "")
                if isinstance(result_text, str) and result_text:
                    # Only use result text if we didn't already stream it
                    if not collected:
                        print(result_text, end="", flush=True)
                    collected.append(result_text)

        # Ensure a trailing newline after streamed output
        if collected:
            print(flush=True)

        # Drain stderr
        assert proc.stderr is not None
        stderr_out = proc.stderr.read()
        if stderr_out:
            print(stderr_out, file=sys.stderr)

        proc.wait()
    except (KeyboardInterrupt, Exception):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise

    return "".join(collected), proc.returncode


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace, claude_args: list[str]) -> int:
    parts_dir = pathlib.Path(args.parts_dir) if args.parts_dir else DEFAULT_PARTS_DIR
    all_parts = load_parts(parts_dir)
    active_parts = resolve_active_parts(all_parts, args.parts)

    user_prompt = resolve_prompt(args)
    if not user_prompt.strip():
        print("Error: empty prompt.", file=sys.stderr)
        return 1

    full_prompt = assemble_prompt(user_prompt, active_parts)

    if args.dry_run:
        print(full_prompt)
        return 0

    # -- Permission / tool-approval flags --------------------------------
    env = os.environ.copy()

    if args.safe:
        # Safe mode: use --allowedTools instead of skipping permissions
        for tool in DEFAULT_ALLOWED_TOOLS:
            if f"--allowedTools={tool}" not in claude_args:
                claude_args += ["--allowedTools", tool]
    else:
        # Default: yolo mode
        if "--dangerously-skip-permissions" not in claude_args:
            claude_args = ["--dangerously-skip-permissions"] + claude_args
        env["IS_SANDBOX"] = "1"

    cmd_base = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--effort", "high"] + claude_args

    for i in range(1, args.max_iterations + 1):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Iteration {i}/{args.max_iterations}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        collected_text, returncode = _run_streaming(cmd_base, full_prompt, env)

        # Strict signal check: exact match on a stripped line
        if any(
            line.strip() == COMPLETION_SIGNAL
            for line in collected_text.splitlines()
        ):
            print(f"\n✓ Completed after {i} iteration(s).", file=sys.stderr)
            return 0

        if returncode != 0:
            print(
                f"\n⚠ claude exited with code {returncode} on iteration {i}.",
                file=sys.stderr,
            )

    print(
        f"\n✗ Reached max iterations ({args.max_iterations}) without completion.",
        file=sys.stderr,
    )
    return 1


def main():
    # Split on -- to separate our args from claude passthrough args
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        our_argv = argv[:split]
        claude_argv = argv[split + 1:]
    else:
        our_argv = argv
        claude_argv = []

    parser = build_parser()
    args = parser.parse_args(our_argv)
    sys.exit(run(args, claude_argv))


if __name__ == "__main__":
    main()
