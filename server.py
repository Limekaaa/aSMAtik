from mesa.visualization import ModularServer, CanvasGrid, ChartModule
from src.model import RobotMission
from src.agents import GreenRobot, YellowRobot, RedRobot
from src.objects import WasteAgent, RadioactivityAgent, WasteDisposalZoneAgent

def agent_draw(agent):
    """Dessiner les agents sur la grille"""
    if agent.__class__.__name__ == 'GreenRobot':
        return {"color": "green", "size": 0.8}
    elif agent.__class__.__name__ == 'YellowRobot':
        return {"color": "yellow", "size": 0.8}
    elif agent.__class__.__name__ == 'RedRobot':
        return {"color": "red", "size": 0.8}
    elif agent.__class__.__name__ == 'WasteAgent':
        if agent.waste_type == 'green':
            return {"color": "#00ff00", "size": 0.5}
        elif agent.waste_type == 'yellow':
            return {"color": "#ffff00", "size": 0.5}
        else:  # red
            return {"color": "#ff0000", "size": 0.5}
    elif agent.__class__.__name__ == 'WasteDisposalZoneAgent':
        return {"color": "purple", "size": 0.9}
    elif agent.__class__.__name__ == 'RadioactivityAgent':
        # Optionnel: dessiner la radioactivité
        gray = int(255 * agent.radioactivity_level)
        return {"color": f"rgb({gray}, {gray}, {gray})", "size": 0.3}
    return {"color": "white", "size": 0.5}

# Créer la visualisation
grid = CanvasGrid(agent_draw, 20, 10, 500, 500)
chart = ChartModule(
    [{"Label": "Green Waste", "Color": "green"},
     {"Label": "Yellow Waste", "Color": "yellow"},
     {"Label": "Red Waste", "Color": "red"},
     {"Label": "Disposed Waste", "Color": "purple"}],
    data_collector_name="datacollector"
)

# Créer le serveur
server = ModularServer(
    RobotMission,
    [grid, chart],
    "Robot Mission Simulation",
    {
        "width": 20,
        "height": 10,
        "num_green_robots": 2,
        "num_yellow_robots": 2,
        "num_red_robots": 1,
        "num_initial_waste": 15
    }
)

if __name__ == "__main__":
    server.port = 8521
    server.launch()