import argparse

from src.experiments import run_experiments
from src.plots import make_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Run the INDENG 164 LLM routing optimization pipeline.")
    parser.add_argument("--data", default="data/routerbench.csv", help="Path to locked routerbench CSV.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated tables, figures, and solutions.")
    parser.add_argument("--skip-a1", action="store_true", help="Skip A1 single-shot MILP grid.")
    parser.add_argument("--skip-a2", action="store_true", help="Skip A2 cascade MILP grid.")
    parser.add_argument("--skip-a3", action="store_true", help="Skip A3 robust cascade MILP.")
    parser.add_argument("--only-plots", action="store_true", help="Regenerate figures from existing output CSVs.")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Per-solve solver time limit in seconds.")
    parser.add_argument("--max-cascades", type=int, default=250, help="Maximum global cascade candidates.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.only_plots:
        make_all_plots(args.output_dir)
        print(f"Regenerated figures under {args.output_dir}/figures")
        return
    result = run_experiments(
        data_path=args.data,
        output_dir=args.output_dir,
        skip_a1=args.skip_a1,
        skip_a2=args.skip_a2,
        skip_a3=args.skip_a3,
        time_limit=args.time_limit,
        max_cascades=args.max_cascades,
    )
    print("Finished routing experiments")
    print(f"Output directory: {result['output_dir']}")
    print(f"A1 grid points: {result['a1_count']}")
    print(f"A2 grid points: {result['a2_count']}")
    print(f"A3 attempts: {result['a3_count']}")


if __name__ == "__main__":
    main()
