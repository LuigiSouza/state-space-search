from __future__ import annotations
import matplotlib.pyplot as plt
import time

import src.Grid as Grid


def main():

    grid = [
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
    ssg = Grid.TSG(grid)
    ssg.create_graph()

    global_edges = set()
    for x in ssg.vertices:
        vert = ssg.vertices[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            destiny_str = str(destiny.x) + "," + str(destiny.y)
            if (
                destiny.x,
                destiny.y,
                vert.x,
                vert.y,
            ) in global_edges:
                continue
            global_edges.add((vert.x, vert.y, destiny.x, destiny.y))
            plt.plot([vert.x, destiny.x], [vert.y, destiny.y], "g-", lw=0.5)
    plot_grid = [
        [[y * 255, y * 255, y * 255] if type(y) == int else [255, 0, 0] for y in x]
        for x in ssg.grid
    ]
    result, _ = ssg.a_grid_search((0, 0), (7, 0))
    plt.imshow(plot_grid)
    for res in result:
        plt.plot([res[0]], [res[1]], "go", lw=0.5)
    plt.show()

    return

    grid = create_grid("maps/Denver_2_1024.map")
    # grid = [
    #     [1, 1, 1, 1, 1, 1, 0, 1],
    #     [1, 0, 0, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 0, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1],
    # ]
    vertices = create_ssg(grid)
    global_goals, local_goals = create_tsg(vertices, grid)

    start = time.time()
    plot_grid = [
        [[y * 255, y * 255, y * 255] if type(y) == int else [255, 0, 0] for y in x]
        for x in grid
    ]
    plt.imshow(plot_grid)
    total_edges = set()
    global_edges = set()
    local_edges = set()
    for x in global_goals:
        vert = global_goals[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            if (destiny.x, destiny.y, vert.x, vert.y) in total_edges:
                continue
            total_edges.add((vert.x, vert.y, destiny.x, destiny.y))
    for x in global_goals:
        vert = global_goals[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            destiny_str = str(destiny.x) + "," + str(destiny.y)
            if (
                destiny.x,
                destiny.y,
                vert.x,
                vert.y,
            ) in global_edges or destiny_str in local_goals:
                continue
            global_edges.add((vert.x, vert.y, destiny.x, destiny.y))
            plt.plot([vert.x, destiny.x], [vert.y, destiny.y], "g-", lw=0.5)
    for x in local_goals:
        vert = local_goals[x]
        for edge in vert.edges:
            destiny = vert.edges[edge].destiny
            if (destiny.x, destiny.y, vert.x, vert.y) in local_edges:
                continue
            local_edges.add((vert.x, vert.y, destiny.x, destiny.y))
            # plt.plot([vert.x, destiny.x], [vert.y, destiny.y], "g-", lw=0.5)
    print(
        f"Global edges: {len(global_edges)} = {len(total_edges)} - {len(local_edges)}"
    )
    print(f"Global Goals: {len(global_goals)} = {len(vertices)} - {len(local_goals)}")
    print(f"Plot created in {time.time() - start} seconds")
    plt.show()


if __name__ == "__main__":
    main()
