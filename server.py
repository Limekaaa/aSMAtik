# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import mesa
import solara
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from mesa.visualization.utils import update_counter

# Import the local model.py
from src.model import RobotMission


def agent_portrayal(agent):
    portrayal = {}

    # Robots
    if hasattr(agent, "robot_type"):
        portrayal["marker"] = "o"
        portrayal["size"] = 40
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
    "width": 20,
    "height": 10,
    "num_green_robots": {
        "type": "SliderInt",
        "value": 2,
        "min": 1,
        "max": 10,
        "step": 1,
        "label": "Green robots",
    },
    "num_yellow_robots": {
        "type": "SliderInt",
        "value": 2,
        "min": 1,
        "max": 10,
        "step": 1,
        "label": "Yellow robots",
    },
    "num_red_robots": {
        "type": "SliderInt",
        "value": 1,
        "min": 1,
        "max": 5,
        "step": 1,
        "label": "Red robots",
    },
    "num_initial_waste": {
        "type": "SliderInt",
        "value": 15,
        "min": 5,
        "max": 50,
        "step": 1,
        "label": "Initial waste",
    },
}

# Create initial model instance
model1 = RobotMission()

SpaceGraph = make_space_component(agent_portrayal)
WastePlot = make_plot_component([
    "Green Waste",
    "Yellow Waste",
    "Red Waste",
    "Disposed Waste"
])

#Create the Dashboard
page = SolaraViz(
    model1,
    components=[SpaceGraph, WastePlot],
    model_params=model_params,
    name="aSMAtik",
)
# This is required to render the visualization in the Jupyter notebook
page
# to start : "solara run server.py"