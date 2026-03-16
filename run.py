from src.model import RobotMission
NUM_STEP = 100

model = RobotMission(
    width=20,
    height=10,
    num_green_robots=2,
    num_yellow_robots=2,
    num_red_robots=1,
    num_initial_waste=15
)

for i in range(NUM_STEP):
    model.step()
    print(f"Step {i+1}: {model._count_total_waste()} waste remaining")

data = model.datacollector.get_model_vars_dataframe()
print(data)