#!/usr/bin/env python3
"""
Ralph Wiggum Method as in https://sidbharath.com/blog/ralph-wiggum-claude-code/
run Claude Code in a loop until completion.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Modular prompt parts registry
# ---------------------------------------------------------------------------

@dataclass
class PromptPart:
    name: str           # CLI flag: "verify" → --verify / --no-verify
    description: str    # Shown in --help
    default: bool       # Enabled by default?
    content: str        # Prompt text (may contain {format} vars)


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

PROMPT_PARTS: list[PromptPart] = [
    PromptPart(
        name="small-changes",
        description="Keep each iteration focused and small",
        default=True,
        content=(
            "You are one iteration of agentic loop to complete this task. "
            "Keep changes small and focused. Each iteration should accomplish "
            "one logical unit of work. Do not try to do everything at once."
        ),
    ),
    PromptPart(
        name="tests",
        description="Run tests and linting before declaring completion",
        default=False,
        content=(
            "Before declaring completion, run the full test suite and fix any "
            "failures. Run type checking and linting and fix any issues."
        ),
    ),
    PromptPart(
        name="progress",
        description="Maintain an iteration progress log",
        default=True,
        content=(
            "Maintain a progress log in `{progress_file}`. At the start of "
            "each iteration, read the file to understand prior work. At the "
            "end, append a concise summary of what you accomplished this "
            "iteration."
        ),
    ),
    PromptPart(
        name="checklist",
        description="Require explicit completion criteria check",
        default=True,
        content=(
            "Do not output the complete signal when you only finish a single task. "
            "Every task in the prompt should be finished, with all verfication done and logged, "
            "and everything that needs to be run should be run. "
            "DO NOT output the signal unless ALL criteria are met, "
            "and there is nothing at all to do. "
            "If you are unsure, finish your work without completing, "
            "and document in the progress file that the next iteration should review everything."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompt(
    user_prompt: str,
    enabled_parts: dict[str, bool],
    completion_signal: str,
    progress_file: str,
) -> str:
    fmt = dict(
        completion_signal=completion_signal,
        progress_file=progress_file,
    )

    sections = [user_prompt.strip()]

    active = [p for p in PROMPT_PARTS if enabled_parts.get(p.name, p.default)]
    lines = ["", "## Iteration Protocol"]
    for part in active:
        lines.append(f"- {part.content.format(**fmt)}")
    lines.append(
        f'- When ALL completion criteria are met, output the exact line: {completion_signal}'
    )
    sections.append("\n".join(lines))

    return "\n".join(sections)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Claude Code in a loop until completion.",
        epilog="Anything after -- is passed directly to `claude -p`.",
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
        default=30,
        help="Maximum loop iterations (default: 30).",
    )
    p.add_argument(
        "--completion-signal",
        default="COMPLETE",
        help='String that signals completion (default: "COMPLETE").',
    )
    p.add_argument(
        "--progress-file",
        default="PROGRESS.md",
        help="Path to the progress log file (default: PROGRESS.md).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assembled prompt and exit.",
    )
    p.add_argument(
        "--list-parts",
        action="store_true",
        help="List all available prompt parts and exit.",
    )
    p.add_argument(
        "--yolo",
        action="store_true",
        help=(
            "Use --dangerously-skip-permissions with IS_SANDBOX=1 "
            "(for containerised / throwaway environments)."
        ),
    )
    p.add_argument(
        "--no-allow-defaults",
        action="store_true",
        help=(
            "Do not inject the default --allowedTools set. "
            "By default, Edit/Write/Bash/Agent/NotebookEdit/WebFetch/WebSearch "
            "are pre-approved for non-interactive use."
        ),
    )

    # Auto-generate --{name} / --no-{name} for each prompt part
    parts_group = p.add_argument_group("prompt parts (each has --X / --no-X)")
    for part in PROMPT_PARTS:
        parts_group.add_argument(
            f"--{part.name}",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"{'[ON] ' if part.default else '[OFF] '}{part.description}",
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


def resolve_enabled_parts(args: argparse.Namespace) -> dict[str, bool]:
    enabled = {}
    for part in PROMPT_PARTS:
        flag_val = getattr(args, part.name.replace("-", "_"), None)
        if flag_val is not None:
            enabled[part.name] = flag_val
        else:
            enabled[part.name] = part.default
    return enabled

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
    return "".join(collected), proc.returncode


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace, claude_args: list[str]) -> int:
    enabled = resolve_enabled_parts(args)

    if args.list_parts:
        for part in PROMPT_PARTS:
            state = "ON" if enabled[part.name] else "OFF"
            print(f"  [{state}]  --{part.name:20s} {part.description}")
        return 0

    user_prompt = resolve_prompt(args)
    if not user_prompt.strip():
        print("Error: empty prompt.", file=sys.stderr)
        return 1

    full_prompt = assemble_prompt(
        user_prompt, enabled, args.completion_signal, args.progress_file,
    )

    if args.dry_run:
        print(full_prompt)
        return 0

    # -- Permission / tool-approval flags --------------------------------
    env = os.environ.copy()

    if args.yolo:
        if "--dangerously-skip-permissions" not in claude_args:
            claude_args = ["--dangerously-skip-permissions"] + claude_args
        env["IS_SANDBOX"] = "1"
    elif not args.no_allow_defaults:
        # Inject a broad-but-safe --allowedTools set so claude -p can
        # actually do work without interactive approval prompts.
        for tool in DEFAULT_ALLOWED_TOOLS:
            if f"--allowedTools={tool}" not in claude_args:
                claude_args += ["--allowedTools", tool]

    cmd_base = ["claude", "-p", "--output-format", "stream-json", "--verbose"] + claude_args

    for i in range(1, args.max_iterations + 1):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Iteration {i}/{args.max_iterations}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        collected_text, returncode = _run_streaming(cmd_base, full_prompt, env)

        # Strict signal check: exact match on a stripped line
        if any(
            line.strip() == args.completion_signal
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
