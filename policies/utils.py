# Group 11
# Created 16-03-2026
# Quentin GUIGNARD, Maxime HANUS, Thomas PEDENAUD

def waste_here(model, pos, wtype):
    cell_contents = model.grid.get_cell_list_contents([pos])
    return any(
        hasattr(obj, "waste_type") and obj.waste_type == wtype
        for obj in cell_contents
    )


def get_accessible_neighbors(model, agent, pos):
    x, y = pos
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    neighbors = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < model.width and 0 <= ny < model.height:
            zone = model._get_zone(nx)

            if agent.can_access_zone(zone):
                neighbors.append(((nx, ny), (dx, dy)))

    return neighbors