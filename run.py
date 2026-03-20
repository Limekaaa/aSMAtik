# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

from src.model import RobotMission

if __name__ == "__main__":
    NUM_STEPS = 200
    
    print("=" * 60)
    print("ROBOT MISSION SIMULATION - Execution")
    print("=" * 60)
    
    # Create and run model
    model = RobotMission(
        width=20,
        height=10,
        num_green_robots=2,
        num_yellow_robots=2,
        num_red_robots=1,
        num_initial_waste=15
    )
    
    print(f"\nRunning {NUM_STEPS} simulation steps...\n")
    
    for i in range(NUM_STEPS):
        model.step()
        if (i + 1) % 50 == 0:
            total = model._count_total_waste()
            disposed = model._count_disposed_waste()
            print(f"Step {i+1:3d}: {total:2d} waste remaining | {disposed:2d} disposed")
    
    # Get and display data
    data = model.datacollector.get_model_vars_dataframe()
    data.to_csv('simulation_data.csv')
    
    print(f"\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total waste in environment: {data['Total Waste'].iloc[-1]}")
    print(f"Total waste disposed: {data['Disposed Waste'].iloc[-1]}")
    print(f"Data saved to: simulation_data.csv")
    print("=" * 60)