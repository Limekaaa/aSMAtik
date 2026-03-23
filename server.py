# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import mesa
import solara
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from mesa.visualization.utils import update_counter
import matplotlib.pyplot as plt

# Import the local model.py
from src.model import RobotMission


def agent_portrayal(agent):
    portrayal = {}

    # Robots
    if hasattr(agent, "robot_type"):
        portrayal["marker"] = "o"
        portrayal["size"] = 50
        portrayal["alpha"] = 1

        colors = {
            "green": "green",
            "yellow": "gold",
            "red": "red"
        }
        portrayal["color"] = colors[agent.robot_type]

    # Waste
    elif hasattr(agent, "waste_type"):
        portrayal["marker"] = "^"
        portrayal["size"] = 10
        portrayal["alpha"] = 1

        colors = {
            "green": "green",
            "yellow": "gold",
            "red": "red"
        }
        portrayal["color"] = colors[agent.waste_type]

    # Disposal zone
    elif agent == agent.model.waste_disposal_zone:
        portrayal["marker"] = "X"
        portrayal["size"] = 80
        portrayal["color"] = "black"
        portrayal["alpha"] = 1

    # Radioactivity
    elif hasattr(agent, "radioactivity_level"):

        if agent.zone == "z1":
            portrayal["color"] = "green"
        elif agent.zone == "z2":
            portrayal["color"] = "yellow"
        else:
            portrayal["color"] = "red"

        portrayal["marker"] = "s"
        portrayal["size"] = 300
        portrayal["alpha"] = 0.2

    return portrayal


model_params = {
    "num_green_robots": {
        "type": "SliderInt",
        "value": 1,
        "label": "Number of green agents:",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    "num_yellow_robots": {
        "type": "SliderInt",
        "value": 1,
        "label": "Number of yellow agents:",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    "num_red_robots": {
        "type": "SliderInt",
        "value": 1,
        "label": "Number of red agents:",
        "min": 0,
        "max": 10,
        "step": 1,
    },
    "num_initial_waste": {
        "type": "SliderInt",
        "value": 1,
        "label": "Number of waste:",
        "min": 0,
        "max": 10,
        "step": 1,
    }
}
# Configuration des graphiques Mesa
SpaceGraph = make_space_component(agent_portrayal)
WastePlot = make_plot_component({
    "Green Waste": "#00AA00",
    "Yellow Waste": "#FFD700", 
    "Red Waste": "#FF4444",
    "Disposed Waste": "#888888"
})

@solara.component
def RobotStatusPanel(model_inst):
    update_counter.get()

    with solara.Card("État des Robots"):
        with solara.Column():            
            # Mission complète ? Utilise ta condition d'arrêt [cite: 82]
            if model_inst.is_done():
                solara.Success("Mission accomplie : Zone décontaminée !")

            for i, robot in enumerate(getattr(model_inst, "robots", [])):
                inv = robot.inventory
                # On affiche la dernière action pour voir ta policy en direct [cite: 26, 29]
                action = getattr(robot, "last_action", {})
                act_name = action.get("type", "Idle") if isinstance(action, dict) else str(action)
                
                color = {"green": "green", "yellow": "gold", "red": "red"}.get(robot.robot_type, "gray")

                with solara.Row(style=f"border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 5px;"):
                    solara.Text(f"Robot {robot.robot_type.upper()} #{i+1}")
                    solara.Text(f"Position: {robot.pos}")
                    solara.Text(f"Inventory: {len(inv)} items")
                    solara.Text(f"Last Action: {act_name}")

# On crée l'instance unique ici
model1 = RobotMission(width=20, height=10, num_initial_waste=model_params["num_initial_waste"]["value"] , num_green_robots=model_params["num_green_robots"]["value"], num_yellow_robots=model_params["num_yellow_robots"]["value"], num_red_robots=model_params["num_red_robots"]["value"])
# On lance SolaraViz avec l'INSTANCE
page = SolaraViz(
    model1, 
    components=[SpaceGraph, WastePlot, RobotStatusPanel],
    model_params=model_params,
    name="aSMAtik"
)