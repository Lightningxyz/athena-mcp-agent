import argparse
import json
import os
from agent.core import Agent
from agent.autofix import AutoFixAgent
from agent.review import ReviewAgent
from eval.runner import run_eval, load_history, save_history, enforce_gates
from config import EVAL_HISTORY_FILE
from utils.logger import print_trace

def main() -> int:
    parser = argparse.ArgumentParser(description="Athena: Deterministic MCP Developer Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    debug_parser = subparsers.add_parser("debug")
    debug_parser.add_argument("query")
    debug_parser.add_argument("--path", default=".")
    
    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("path")

    autofix_parser = subparsers.add_parser("autofix")
    autofix_parser.add_argument("query")
    autofix_parser.add_argument("--path", default=".")
    autofix_parser.add_argument("--apply", action="store_true")
    autofix_parser.add_argument("--max-iterations", type=int, default=4)
    autofix_parser.add_argument("--plan-candidates", type=int, default=3)
    autofix_parser.add_argument("--approve-high-risk", action="store_true")
    autofix_parser.add_argument("--git", action="store_true", help="Enable git-aware workflow features.")
    autofix_parser.add_argument("--create-branch", action="store_true", help="Create a work branch when --git is enabled.")
    autofix_parser.add_argument("--auto-commit", action="store_true", help="Auto-commit successful patch iterations when --git is enabled.")
    autofix_parser.add_argument(
        "--validate-cmd",
        action="append",
        default=[],
        help="Validation command to run after applying edits. Can be passed multiple times.",
    )

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--tasks", default="eval/tasks.json")

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--path", default=".")
    review_parser.add_argument("--focus", default="")
    review_parser.add_argument("--staged", action="store_true")
    review_parser.add_argument("--base-ref", default=None)

    session_parser = subparsers.add_parser("session")
    session_parser.add_argument("--path", default=".")
    session_parser.add_argument("--apply", action="store_true")
    session_parser.add_argument("--git", action="store_true")
    session_parser.add_argument("--approve-high-risk", action="store_true")
    session_parser.add_argument("--plan-candidates", type=int, default=3)
    
    args = parser.parse_args()
    
    from utils.spinner import Spinner
    with Spinner("Initializing agent...") as spinner:
        agent = Agent(spinner=spinner)
        
        if args.command == "debug":
            result = agent.run("debug", args.query, args.path)
            print("\n--- Final Answer ---")
            print(result)
        elif args.command == "explain":
            result = agent.run("explain", f"Explain the codebase at {args.path}", args.path)
            print("\n--- Final Answer ---")
            print(result)
        elif args.command == "autofix":
            fixer = AutoFixAgent(agent)
            report = fixer.run(
                query=args.query,
                path=args.path,
                apply=args.apply,
                max_iterations=max(1, args.max_iterations),
                plan_candidates=max(1, args.plan_candidates),
                validation_commands=args.validate_cmd or None,
                approve_high_risk=bool(args.approve_high_risk),
                git_mode=bool(args.git),
                create_branch=bool(args.create_branch),
                auto_commit=bool(args.auto_commit),
            )
            print("\n--- AutoFix Report ---")
            print(json.dumps(report, indent=2))
            if args.apply:
                print(f"\nApplied mode enabled for workspace: {os.path.abspath(args.path)}")
        elif args.command == "evaluate":
            summary = run_eval(args.tasks)
            history = load_history(EVAL_HISTORY_FILE)
            previous = history[-1] if history else {}
            enforce_gates(summary, previous)
            history.append(summary)
            history = history[-50:]
            save_history(EVAL_HISTORY_FILE, history)
            print("\n--- Eval Summary ---")
            print(json.dumps(summary, indent=2))
        elif args.command == "review":
            reviewer = ReviewAgent(agent)
            report = reviewer.run(
                path=args.path,
                focus=args.focus,
                staged=bool(args.staged),
                base_ref=args.base_ref,
            )
            print("\n--- Review Report ---")
            print(json.dumps(report, indent=2))
        elif args.command == "session":
            workspace = os.path.abspath(args.path)
            fixer = AutoFixAgent(agent)
            print("\n--- Athena Interactive Session ---")
            reviewer = ReviewAgent(agent)
            print("Type `debug: <query>`, `explain`, `autofix: <goal>`, `review`, or `exit`.")
            while True:
                try:
                    raw = input("athena> ").strip()
                except EOFError:
                    break
                if not raw:
                    continue
                if raw.lower() in {"exit", "quit"}:
                    break
                if raw.lower().startswith("debug:"):
                    query = raw.split(":", 1)[1].strip()
                    result = agent.run("debug", query, workspace)
                    print("\n" + result + "\n")
                    continue
                if raw.lower() == "explain":
                    result = agent.run("explain", f"Explain the codebase at {workspace}", workspace)
                    print("\n" + result + "\n")
                    continue
                if raw.lower().startswith("autofix:"):
                    query = raw.split(":", 1)[1].strip()
                    report = fixer.run(
                        query=query,
                        path=workspace,
                        apply=bool(args.apply),
                        max_iterations=3,
                        plan_candidates=max(1, args.plan_candidates),
                        validation_commands=None,
                        approve_high_risk=bool(args.approve_high_risk),
                        git_mode=bool(args.git),
                        create_branch=False,
                        auto_commit=False,
                    )
                    print("\n" + json.dumps(report, indent=2) + "\n")
                    continue
                if raw.lower().startswith("review"):
                    focus = ""
                    if ":" in raw:
                        focus = raw.split(":", 1)[1].strip()
                    report = reviewer.run(path=workspace, focus=focus, staged=False, base_ref=None)
                    print("\n" + json.dumps(report, indent=2) + "\n")
                    continue
                print("Unrecognized command. Use `debug: ...`, `explain`, `autofix: ...`, `review`, or `exit`.")
        else:
            return 1

    trace_file = agent.logger.export_jsonl()
    print(f"\nTrace exported to: {trace_file}")
    print_trace(agent.logger)
    
    return 0
