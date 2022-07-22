from __future__ import annotations
import matplotlib.pyplot as plt
from time import time

import src.Grid as Grid


def main():

    grid = [
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]

    tsg = Grid.TSG(grid)
    # tsg.create_graph()
    tsg.create_from_file("maps/Paris_0_1024.map")

    plot_grid = [
        [[y * 255, y * 255, y * 255] if type(y) == int else [255, 0, 0] for y in x]
        for x in tsg.grid
    ]
    start_time = time()
    result, weight, close = tsg.a_grid_graph_search((0, 0), (550, 515))
    print(f"A* with Visibility Graph finished in {time() - start_time} seconds")
    print("Weight: ", weight)

    plt.imshow(plot_grid)

    x = [i[0] for i in close if i not in result]
    y = [i[1] for i in close if i not in result]
    plt.plot(x, y, "ro", lw=0.5)
    x = [x for x, _ in result]
    y = [y for _, y in result]
    plt.plot(x, y, "g-", lw=1)

    plt.show()


if __name__ == "__main__":
    main()
