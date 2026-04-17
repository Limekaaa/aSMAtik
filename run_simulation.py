# Group 11
# Created 27-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import argparse
import sys
import src.workspace as ws
from src.model import RobotMission


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
        choices=['baseline_policy', 'random_policy', 'baseline_communication', 'communication_v2', 'RL_baseline_RL_train'],
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
        help='Number of simulation steps (default: 200)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output',
        type=str,
        default='simulation_data.csv',
        help='Output CSV file name (default: simulation_data.csv)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print step-by-step information during simulation'
    )
    
    args = parser.parse_args()
    
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
    print(f"  - Simulation steps: {args.steps}")
    if args.seed:
        print(f"  - Random seed:  {args.seed}")
    print(f"\nOutput file:      {args.output}")
    print("\n" + "=" * 70)
    print("SIMULATION EXECUTION")
    print("=" * 70 + "\n")
    
    try:
        # Set the policy in workspace
        ws.POLICY = args.policy
        
        # Create model with specified parameters
        model = RobotMission(
            width=args.width,
            height=args.height,
            num_green_robots=args.green_robots,
            num_yellow_robots=args.yellow_robots,
            num_red_robots=args.red_robots,
            num_initial_waste=args.initial_waste,
            seed=args.seed
        )
        
        # Run simulation
        print(f"Running {args.steps} simulation steps...\n")
        
        for i in range(args.steps):
            model.step()
            
            if args.verbose and (i + 1) % 50 == 0:
                total = model._count_total_waste()
                disposed = model._count_disposed_waste()
                print(f"Step {i+1:4d}/{args.steps}: "
                      f"Waste remaining: {total:2d} | "
                      f"Waste disposed: {disposed:2d}")
            elif (i + 1) % 50 == 0:
                print(f"  Step {i+1:4d}/{args.steps} completed")
        
        # Collect and display results
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        
        data = model.datacollector.get_model_vars_dataframe()
        
        final_total_waste = data['Total Waste'].iloc[-1]
        final_disposed_waste = data['Disposed Waste'].iloc[-1]
        initial_total = args.initial_waste
        collection_rate = ((initial_total - final_total_waste) / initial_total * 100) if initial_total > 0 else 0
        
        print("\nWaste Management:")
        print(f"  - Initial waste:       {initial_total}")
        print(f"  - Total waste in env:  {final_total_waste}")
        print(f"  - Total waste disposed:{final_disposed_waste}")
        print(f"  - Collection rate:     {collection_rate:.1f}%")
        
        print(f"\nData saved to: {args.output}")
        data.to_csv(args.output)
        
        print("=" * 70 + "\n")
        print("✓ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
