"""run_evaluation.py — FinWiki evaluation harness dispatcher

Usage:
  python run_evaluation.py --version v4 --all-conditions
  python run_evaluation.py --version v3 --all-conditions
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="FinWiki evaluation harness runner")
    parser.add_argument("--version", required=True, choices=["v1", "v2", "v3", "v4"],
                        help="Harness version to run")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run all retrieval conditions")
    args = parser.parse_args()

    if args.version == "v4":
        from pipeline.eval_harness_v4 import main as run
    elif args.version == "v3":
        from pipeline.eval_harness_v3 import main as run
    elif args.version == "v2":
        from pipeline.eval_harness_v2 import main as run
    elif args.version == "v1":
        from pipeline.eval_harness import main as run
    else:
        print(f"Unknown version: {args.version}", file=sys.stderr)
        sys.exit(1)

    run()


if __name__ == "__main__":
    main()
