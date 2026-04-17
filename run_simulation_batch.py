from run_simulation import main
import argparse
import os 
import src.workspace as ws
if __name__ == "__main__":

    ws.train = False

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
        default='batch_simulation',
        help='Output folder containing the simulation data (default: batch_simulation)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print step-by-step information during simulation'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of simulations to run in batch (default: 5)'
    )
    
    args = parser.parse_args()
    n_sim = args.batch_size
    os.makedirs(args.output, exist_ok=True)


    for i in range(n_sim):
        args = parser.parse_args()

        print(f"Running simulation {i+1}/{args.batch_size} with seed {i}")
        args_dict = vars(args)
        args_dict["output"] = os.path.join(args.output, f"simulation_{i+1}.csv")
        args_dict["seed"] = i  # Set a different seed for each simulation
        args = argparse.Namespace(**args_dict)
        main(args)

    args = parser.parse_args()
    print(f"Completed {args.batch_size} simulations. Data saved in folder: {args.output}")

    

