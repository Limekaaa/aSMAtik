# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import solara
import threading
import time
from matplotlib.figure import Figure
from src.model import RobotMission

# Reactive state for the model
model_state: solara.Reactive[RobotMission] = solara.reactive(None)
step_counter: solara.Reactive[int] = solara.reactive(0)
running: solara.Reactive[bool] = solara.reactive(True)


def simulation_loop():
    """Background thread that runs the simulation."""
    model = RobotMission(
        width=20,
        height=10,
        num_green_robots=2,
        num_yellow_robots=2,
        num_red_robots=1,
        num_initial_waste=15
    )
    model_state.value = model
    
    step = 0
    while running.value:
        try:
            model.step()
            step += 1
            step_counter.value = step
            # Delay to see each tick clearly (500ms per step)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in simulation step {step}: {e}")
            time.sleep(1)


@solara.component
def Page():
    """Main Solara page component."""
    # Initialize simulation on first render
    if model_state.value is None:
        thread = threading.Thread(target=simulation_loop, daemon=True)
        thread.start()
    
    model = model_state.value
    step = step_counter.value  # Use this to trigger re-renders
    
    with solara.Column(style="width: 100%; max-width: 1400px; margin: 0 auto;"):
        solara.Markdown("# Robot Mission - Waste Disposal System")
        
        if model is None:
            solara.Markdown("**Initializing simulation...**")
        else:
            solara.Markdown(f"**Step: {step}**")
            
            with solara.Row():
                with solara.Column(style="flex: 1;"):
                    WasteChart(model, step)
                with solara.Column(style="flex: 1;"):
                    DisposalChart(model, step)
            
            with solara.Row():
                with solara.Column(style="flex: 1;"):
                    Statistics(model)
                with solara.Column(style="flex: 1;"):
                    GridVisualization(model, step)


@solara.component
def WasteChart(model: RobotMission, step: int = 0):
    """Component to display waste levels by type."""
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    
    data = model.datacollector.get_model_vars_dataframe()
    if len(data) > 0:
        steps = range(len(data))
        ax.plot(steps, data['Green Waste'], label='Green', color='green', linewidth=2)
        ax.plot(steps, data['Yellow Waste'], label='Yellow', color='orange', linewidth=2)
        ax.plot(steps, data['Red Waste'], label='Red', color='red', linewidth=2)
        ax.plot(steps, data['Total Waste'], label='Total', color='black', linewidth=2.5, linestyle='--')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Count')
        ax.set_title('Waste in Environment')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Collecting data...', ha='center', va='center')
    
    solara.FigureMatplotlib(fig)


@solara.component
def DisposalChart(model: RobotMission, step: int = 0):
    """Component to display waste disposal progress."""
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    
    data = model.datacollector.get_model_vars_dataframe()
    if len(data) > 0:
        steps = range(len(data))
        ax.plot(steps, data['Disposed Waste'], label='Disposed', color='purple', linewidth=2.5, marker='D')
        ax.fill_between(steps, 0, data['Disposed Waste'], alpha=0.3, color='purple')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Count')
        ax.set_title('Waste Successfully Disposed')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Collecting data...', ha='center', va='center')
    
    solara.FigureMatplotlib(fig)


@solara.component
def Statistics(model: RobotMission, step: int = 0):
    """Component to display simulation statistics."""
    data = model.datacollector.get_model_vars_dataframe()
    
    if len(data) > 0:
        current_waste = int(data['Total Waste'].iloc[-1])
        disposed_waste = int(data['Disposed Waste'].iloc[-1])
        green = int(data['Green Waste'].iloc[-1])
        yellow = int(data['Yellow Waste'].iloc[-1])
        red = int(data['Red Waste'].iloc[-1])
        current_step = len(data) - 1
        
        solara.Markdown(f"""
### Simulation Statistics

**Current Step:** {current_step}

**Waste in Environment:**
- Green: {green}
- Yellow: {yellow}
- Red: {red}
- **Total: {current_waste}**

**Waste Disposed:** {disposed_waste}

**Total Processed:** {int(data['Total Waste'].iloc[0]) - current_waste}
        """)
    else:
        solara.Markdown("**Starting simulation...**")


@solara.component
def GridVisualization(model: RobotMission, step: int = 0):
    """Component to display grid visualization."""
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    
    # Draw background zones
    z1_start, z1_end = model.zone_boundaries['z1']
    z2_start, z2_end = model.zone_boundaries['z2']
    z3_start, z3_end = model.zone_boundaries['z3']
    
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((z1_start, 0), z1_end - z1_start, model.height, 
                            facecolor='#E8F5E9', alpha=0.3, edgecolor='gray', linewidth=2))
    ax.add_patch(Rectangle((z2_start, 0), z2_end - z2_start, model.height, 
                            facecolor='#FFF9C4', alpha=0.3, edgecolor='gray', linewidth=2))
    ax.add_patch(Rectangle((z3_start, 0), z3_end - z3_start, model.height, 
                            facecolor='#FFEBEE', alpha=0.3, edgecolor='gray', linewidth=2))
    
    # Zone labels
    ax.text(z1_start + (z1_end - z1_start) / 2, model.height + 0.5, 'z1 (Low)', 
            ha='center', fontweight='bold', fontsize=9)
    ax.text(z2_start + (z2_end - z2_start) / 2, model.height + 0.5, 'z2 (Medium)', 
            ha='center', fontweight='bold', fontsize=9)
    ax.text(z3_start + (z3_end - z3_start) / 2, model.height + 0.5, 'z3 (High)', 
            ha='center', fontweight='bold', fontsize=9)
    
    # Draw disposal zone
    disposal_pos = model.waste_disposal_zone.pos
    ax.plot(disposal_pos[0], disposal_pos[1], '*', color='#9900FF', 
            markersize=20, label='Disposal Zone', markeredgecolor='black', markeredgewidth=1)
    
    # Draw waste
    for waste in model.waste_pieces:
        waste_colors = {'green': '#00AA00', 'yellow': '#FFAA00', 'red': '#AA0000'}
        ax.plot(waste.pos[0], waste.pos[1], 'o', 
                color=waste_colors.get(waste.waste_type, '#000000'), 
                markersize=6, alpha=0.7)
    
    # Draw robots
    robot_colors = {'GreenRobot': '#00DD00', 'YellowRobot': '#FFDD00', 'RedRobot': '#FF0000'}
    for robot in model.robots:
        robot_type = robot.__class__.__name__
        ax.plot(robot.pos[0], robot.pos[1], 's', 
                color=robot_colors.get(robot_type, '#808080'), markersize=10, 
                markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlim(-1, model.width)
    ax.set_ylim(-1, model.height + 1)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Grid View')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    solara.FigureMatplotlib(fig)