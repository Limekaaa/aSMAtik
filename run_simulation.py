# Group 11
# Created 27-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import argparse
import csv
import random
import sys
import time
from statistics import mean, median
import src.workspace as ws
from src.model import RobotMission


MAX_STEPS_PER_RUN = 1000


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)

    # numpy is optional at runtime; model imports it, but keep runner resilient
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _run_one_simulation(args, seed: int | None, max_steps: int) -> dict:
    ws.POLICY = args.policy
    _seed_everything(seed)

    model = RobotMission(
        width=args.width,
        height=args.height,
        num_green_robots=args.green_robots,
        num_yellow_robots=args.yellow_robots,
        num_red_robots=args.red_robots,
        num_initial_waste=args.initial_waste,
        seed=seed,
    )

    steps_taken = 0
    for _ in range(max_steps):
        model.step()
        steps_taken += 1

        if model.is_done():
            break

    final_total_waste = model._count_total_waste()
    final_disposed_waste = model._count_disposed_waste()

    success = model.is_done()
    reason = "success" if success else "max_steps"

    return {
        "seed": seed,
        "success": success,
        "reason": reason,
        "steps_taken": steps_taken,
        "remaining_waste": final_total_waste,
        "disposed_equivalent": final_disposed_waste,
    }


def _format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _safe_mean(values: list[int]) -> float | None:
    return mean(values) if values else None


def _safe_median(values: list[int]) -> float | None:
    return median(values) if values else None


def _write_summary_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "run",
        "seed",
        "success",
        "reason",
        "steps_taken",
        "remaining_waste",
        "disposed_equivalent",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({"run": i, **row})


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Execute robot mission simulations with custom parameters and policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with baseline policy, default parameters
  python run_simulation.py --policy baseline_policy
  
  # Run with random policy, custom grid and robots
  python run_simulation.py --policy random_policy --width 30 --height 15 --green-robots 3 --yellow-robots 2 --red-robots 1
  
  # Run with communication policy, custom waste and steps
  python run_simulation.py --policy communication_v2 --initial-waste 25 --steps 500
  
  # Run with baseline communication and save with custom name
  python run_simulation.py --policy baseline_communication --output my_simulation.csv --seed 42
        """
    )
    
    # Policy selection
    parser.add_argument(
        '--policy',
        type=str,
        default='baseline_policy',
        choices=['baseline_policy', 'random_policy', 'baseline_communication', 'communication_v2'],
        help='Policy to use for robot behavior (default: baseline_policy)'
    )
    
    # Model parameters
    parser.add_argument(
        '--width',
        type=int,
        default=20,
        help='Grid width (default: 20)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=10,
        help='Grid height (default: 10)'
    )
    parser.add_argument(
        '--green-robots',
        type=int,
        default=2,
        help='Number of green robots (default: 2)'
    )
    parser.add_argument(
        '--yellow-robots',
        type=int,
        default=2,
        help='Number of yellow robots (default: 2)'
    )
    parser.add_argument(
        '--red-robots',
        type=int,
        default=1,
        help='Number of red robots (default: 1)'
    )
    parser.add_argument(
        '--initial-waste',
        type=int,
        default=15,
        help='Initial number of waste pieces (default: 15)'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help=f'Number of simulation steps (capped at {MAX_STEPS_PER_RUN}, default: 200)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of independent runs (default: 1). Use 500 for batch evaluation.'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=MAX_STEPS_PER_RUN,
        help=f'Maximum steps per run (default: {MAX_STEPS_PER_RUN})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility. In batch mode, seeds will be seed+i (default: None)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output',
        type=str,
        default='simulation_data.csv',
        help='Output CSV file name (default: simulation_data.csv)'
    )
    parser.add_argument(
        '--save-summary',
        action='store_true',
        help='In batch mode, save per-run summary CSV to --output'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print step-by-step information during simulation'
    )
    
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    # Enforce step cap
    if args.steps > MAX_STEPS_PER_RUN:
        print(f"Warning: --steps capped to {MAX_STEPS_PER_RUN} (was {args.steps}).")
        args.steps = MAX_STEPS_PER_RUN
    if args.max_steps > MAX_STEPS_PER_RUN:
        print(f"Warning: --max-steps capped to {MAX_STEPS_PER_RUN} (was {args.max_steps}).")
        args.max_steps = MAX_STEPS_PER_RUN
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    
    # Display configuration
    print("=" * 70)
    print("ROBOT MISSION SIMULATION - Configuration")
    print("=" * 70)
    print(f"\nPolicy:           {args.policy}")
    print("\nGrid Parameters:")
    print(f"  - Width:        {args.width}")
    print(f"  - Height:       {args.height}")
    print("\nRobot Fleet:")
    print(f"  - Green robots: {args.green_robots}")
    print(f"  - Yellow robots:{args.yellow_robots}")
    print(f"  - Red robots:   {args.red_robots}")
    print(f"  - Total:        {args.green_robots + args.yellow_robots + args.red_robots}")
    print("\nInitial Conditions:")
    print(f"  - Waste pieces: {args.initial_waste}")
    if args.runs == 1:
        print(f"  - Simulation steps: {args.steps}")
    else:
        print(f"  - Runs:             {args.runs}")
        print(f"  - Max steps / run:  {args.max_steps}")

    if args.seed is not None:
        print(f"  - Seed:         {args.seed}")
    elif args.runs > 1:
        print("  - Seed:         None (non-reproducible batch unless you set --seed)")

    if args.runs == 1:
        print(f"\nOutput file:      {args.output}")
    else:
        if args.save_summary:
            print(f"\nSummary CSV:      {args.output}")
        else:
            print("\nSummary CSV:      (not saved; pass --save-summary to write one)")
    print("\n" + "=" * 70)
    print("SIMULATION EXECUTION")
    print("=" * 70 + "\n")
    
    try:
        if args.runs == 1:
            # Single run (keeps the historical behavior + optional verbose)
            ws.POLICY = args.policy
            _seed_everything(args.seed)

            model = RobotMission(
                width=args.width,
                height=args.height,
                num_green_robots=args.green_robots,
                num_yellow_robots=args.yellow_robots,
                num_red_robots=args.red_robots,
                num_initial_waste=args.initial_waste,
                seed=args.seed,
            )

            print(f"Running up to {args.steps} simulation steps...\n")

            steps_taken = 0
            for i in range(args.steps):
                model.step()
                steps_taken = i + 1

                if args.verbose and (i + 1) % 50 == 0:
                    total = model._count_total_waste()
                    disposed = model._count_disposed_waste()
                    print(
                        f"Step {i+1:4d}/{args.steps}: "
                        f"Waste remaining: {total:2d} | "
                        f"Waste disposed: {disposed:2d}"
                    )
                elif (i + 1) % 50 == 0:
                    print(f"  Step {i+1:4d}/{args.steps} completed")

                if model.is_done():
                    break

            print("\n" + "=" * 70)
            print("FINAL STATISTICS")
            print("=" * 70)

            data = model.datacollector.get_model_vars_dataframe()
            final_total_waste = int(data["Total Waste"].iloc[-1])
            final_disposed_waste = int(data["Disposed Waste"].iloc[-1])
            initial_total = args.initial_waste
            success = model.is_done()
            collection_rate = (
                (initial_total - final_total_waste) / initial_total * 100
                if initial_total > 0
                else 0
            )

            print("\nWaste Management:")
            print(f"  - Success:             {success}")
            print(f"  - Steps executed:      {steps_taken}")
            print(f"  - Initial waste:       {initial_total}")
            print(f"  - Total waste in env:  {final_total_waste}")
            print(f"  - Total disposed (eq): {final_disposed_waste}")
            print(f"  - Collection rate:     {collection_rate:.1f}%")

            print(f"\nData saved to: {args.output}")
            data.to_csv(args.output)

            print("=" * 70 + "\n")
            print("✓ Simulation completed successfully!")

        else:
            # Batch mode
            runs = args.runs
            max_steps = args.max_steps

            # Choose a base seed for non-reproducible batches (still ensures different runs)
            base_seed = args.seed
            if base_seed is None:
                base_seed = int(time.time())
                print(f"Batch base seed (auto): {base_seed}")

            start = time.time()
            rows: list[dict] = []

            for r in range(runs):
                run_seed = base_seed + r
                row = _run_one_simulation(args, run_seed, max_steps=max_steps)
                rows.append(row)

                if (r + 1) % 50 == 0:
                    done = sum(1 for x in rows if x["success"])
                    print(f"  Completed {r+1:4d}/{runs} runs | successes so far: {done}")

            elapsed = time.time() - start

            successes = [x for x in rows if x["success"]]
            failures = [x for x in rows if not x["success"]]

            success_steps = [int(x["steps_taken"]) for x in successes]
            all_steps = [int(x["steps_taken"]) for x in rows]
            remaining = [int(x["remaining_waste"]) for x in rows]
            disposed = [int(x["disposed_equivalent"]) for x in rows]

            success_rate = len(successes) / len(rows) if rows else 0.0

            print("\n" + "=" * 70)
            print("BATCH RESULTS")
            print("=" * 70)
            print(f"Runs:                {runs}")
            print(f"Max steps / run:     {max_steps}")
            print(f"Policy:              {args.policy}")
            print(f"Seed base:           {base_seed}")
            print(f"Elapsed:             {elapsed:.2f}s")

            print("\nSuccess:")
            print(f"  - Successes:        {len(successes)}")
            print(f"  - Failures:         {len(failures)}")
            print(f"  - Success rate:     {_format_pct(success_rate)}")

            avg_steps_success = _safe_mean(success_steps)
            med_steps_success = _safe_median(success_steps)
            avg_steps_all = _safe_mean(all_steps)
            med_steps_all = _safe_median(all_steps)

            print("\nSteps:")
            if avg_steps_success is None:
                print("  - Avg steps (successes): n/a")
                print("  - Median steps (successes): n/a")
            else:
                print(f"  - Avg steps (successes):   {avg_steps_success:.1f}")
                print(f"  - Median (successes):      {med_steps_success:.1f}")
                print(f"  - Min/Max (successes):     {min(success_steps)}/{max(success_steps)}")

            print(f"  - Avg steps (all runs):    {avg_steps_all:.1f}")
            print(f"  - Median (all runs):       {med_steps_all:.1f}")

            print("\nWaste outcomes:")
            print(f"  - Avg remaining waste:     {mean(remaining):.2f}")
            print(f"  - Remaining waste (min/max): {min(remaining)}/{max(remaining)}")
            print(f"  - Avg disposed (eq units): {mean(disposed):.2f}")

            if args.initial_waste > 0:
                avg_collection = mean([min(d, args.initial_waste) / args.initial_waste for d in disposed])
                print(f"  - Avg collection rate:     {_format_pct(avg_collection)}")

            if args.save_summary:
                _write_summary_csv(args.output, rows)
                print(f"\nSummary saved to: {args.output}")

            print("=" * 70 + "\n")
            print("✓ Batch completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
