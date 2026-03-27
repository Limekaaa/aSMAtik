# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

import solara
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter

from src.model import RobotMission
from src.agents import GreenRobot, YellowRobot, RedRobot
from src.objects import WasteAgent, WasteDisposalZoneAgent, RadioactivityAgent


def agent_portrayal(agent):
    """
    Draw only meaningful visible agents.
    Radioactivity is now displayed as a property layer background.
    """

    # Hide radioactivity agents completely
    if isinstance(agent, RadioactivityAgent):
        color = {
            "z1": "lightgreen",
            "z2": "lightyellow",
            "z3": "lightcoral",
        }.get(agent.zone, "lightgray")

        return AgentPortrayalStyle(
            color=color,
            marker="s",
            size=700,   # grand carré de fond
            zorder=0,   # tout en dessous
            alpha=0.5, # léger pour voir les agents dessus
        )

    # Robots
    if isinstance(agent, GreenRobot):
        return AgentPortrayalStyle(
            color="green",
            marker="o",
            size=100,
            zorder=3,
            alpha=1,
        )

    if isinstance(agent, YellowRobot):
        return AgentPortrayalStyle(
            color="gold",
            marker="o",
            size=100,
            zorder=3,
            alpha=1,
        )

    if isinstance(agent, RedRobot):
        return AgentPortrayalStyle(
            color="red",
            marker="o",
            size=100,
            zorder=3,
            alpha=1,
        )

    # Waste
    if isinstance(agent, WasteAgent):
        waste_colors = {
            "green": "#0A8F08",
            "yellow": "#D4B000",
            "red": "#CC2222",
        }
        return AgentPortrayalStyle(
            color=waste_colors.get(agent.waste_type, "gray"),
            marker="^",
            size=80,
            zorder=2,
            alpha=1,
        )

    # Disposal zone
    if isinstance(agent, WasteDisposalZoneAgent):
        return AgentPortrayalStyle(
            color="black",
            marker="X",
            size=200,
            zorder=2,
            alpha=1,
        )

    return AgentPortrayalStyle(color="gray", size=1, alpha=0)




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
        "value": 4,
        "label": "Number of waste:",
        "min": 0,
        "max": 20,
        "step": 1,
    }
}


WastePlot = make_plot_component({
    "Green Waste": "#00AA00",
    "Yellow Waste": "#FFD700",
    "Red Waste": "#FF4444",
    "Disposed Waste": "#888888"
})


@solara.component
def DashboardPanel(model_inst):
    update_counter.get()

    with solara.Column(style="width: 100%; margin-top: 20px; position: relative; z-index: 1;"):
        if model_inst.is_done():
            solara.Success("Mission accomplie : Zone décontaminée !")

        with solara.Columns([1, 1]):
            # Colonne 1 : État des robots
            with solara.Card("État des Robots"):
                robots_list = sorted(getattr(model_inst, "robots", []), key=lambda r: getattr(r, 'unique_id', 0))
                for i, robot in enumerate(robots_list):
                    inv = robot.inventory
                    action = getattr(robot, "last_action", {})
                    act_name = action.get("type", "Idle") if isinstance(action, dict) else str(action)

                    color = {
                        "green": "green",
                        "yellow": "gold",
                        "red": "red",
                    }.get(robot.robot_type, "gray")

                    with solara.Row(
                        style=f"border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 5px;"
                    ):
                        solara.Text(f"{robot.robot_type.upper()} (#{getattr(robot, 'unique_id', i + 1)})")
                        solara.Text(f"Pos: {robot.pos}")
                        solara.Text(f"Inv: {len(inv)}")
                        solara.Text(f"Act: {act_name}")

            # Colonne 2 : Messagerie
            with solara.Card("Messagerie en direct"):
                mailbox = getattr(model_inst, "mailbox", None)
                if mailbox is not None:
                    unread = mailbox._unread_messages
                    read_msg = mailbox._read_messages
                    
                    has_messages = False
                    robots_list = sorted(getattr(model_inst, "robots", []), key=lambda r: getattr(r, 'unique_id', 0))
                    for robot in robots_list:
                        uid = getattr(robot, "unique_id", None)
                        r_unread = unread.get(uid, [])
                        r_read = read_msg.get(uid, [])
                        
                        if r_unread or r_read:
                            has_messages = True
                            color = {"green": "green", "yellow": "gold", "red": "red"}.get(getattr(robot, "robot_type", "gray"), "gray")
                            with solara.Column(style=f"border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 5px; gap: 0px;"):
                                solara.Text(f"Destinataire: {getattr(robot, 'robot_type', 'Robot').upper()} (#{uid})", style="font-weight: bold; font-size: 0.9em;")
                                for m in r_unread:
                                    solara.Text(f" [Non lu] de #{m['sender_id']}: {m['content']}", style="color: #ff4444; font-size: 0.85em;")
                                # Afficher les 3 derniers messages lus maximum
                                for m in r_read[-3:]:
                                    solara.Text(f" [Lu] de #{m['sender_id']}: {m['content']}", style="color: #888888; font-size: 0.85em;")
                    
                    if not has_messages:
                        solara.Text("Aucun message...")
                else:
                    solara.Text("Boîte non trouvée.")

@solara.component
def MessagesPanel(model_inst):
    update_counter.get()

    history = model_inst.mailbox.get_history()

    with solara.Card("Messages échangés", style="max-height: 250px; overflow-y: auto;"):
        if not history:
            solara.Text("Aucun message envoyé.")
        else:
            for msg in reversed(history[-15:]):
                solara.Text(
                    f"Msg #{msg['message_id']} | From {msg['sender_id']} "
                    f"to {msg['recipient_ids']} | {msg['content']}"
                )

model1 = RobotMission(
    width=20,
    height=10,
    num_initial_waste=model_params["num_initial_waste"]["value"],
    num_green_robots=model_params["num_green_robots"]["value"],
    num_yellow_robots=model_params["num_yellow_robots"]["value"],
    num_red_robots=model_params["num_red_robots"]["value"],
)

renderer = SpaceRenderer(model1, backend="matplotlib").render(
    agent_portrayal=agent_portrayal,
)

page = SolaraViz(
    model1,
    renderer,
    components=[
        WastePlot,
        DashboardPanel, 
        MessagesPanel
    ],
    model_params=model_params,
    name="aSMAtik",
)