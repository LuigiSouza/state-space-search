from __future__ import annotations
from random import random
import matplotlib.pyplot as plt
from time import time

import src.Grid as Grid


def generate_bench_points(
    n: int, grid: list[list], file_name: str = "points.txt"
) -> list[tuple[int, int]]:
    bench_mark: list[tuple[int, int]] = []
    max_tries = n * n
    with open(file_name, "w") as f:
        while max_tries and len(bench_mark) < n:
            max_tries -= 1
            x = int(len(grid[0]) * random())
            y = int(len(grid) * random())
            if (x, y) in bench_mark:
                continue
            if grid[y][x] != 0:
                bench_mark.append((x, y))
                f.write(f"{x} {y}\n")
    return bench_mark


def read_bench_points(file_name: str = "points.txt") -> list[tuple[int, int]]:
    bench_mark: list[tuple[int, int]] = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            line = line.split()
            x = int(line[0])
            y = int(line[1])
            bench_mark.append((x, y))
    return bench_mark


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
    tsg.read_file("maps/Berlin_0_1024.map")

    plot_grid = [
        [[y * 255, y * 255, y * 255] if type(y) == int else [255, 0, 0] for y in x]
        for x in tsg.grid
    ]

    bench_mark = read_bench_points()

    start_time = time()
    result, weight, close = tsg.a_grid_graph_search((10, 10), (1020, 1020))
    print(f"A* with Visibility Graph finished in {time() - start_time} seconds")
    print("Weight: ", weight)

    plt.imshow(plot_grid)

    x = [i[0] for i in close if i not in result]
    y = [i[1] for i in close if i not in result]
    plt.plot(x, y, "ro", lw=0.5)
    x = [x for x, _ in result]
    y = [y for _, y in result]
    plt.plot(x, y, "g-", lw=1)
    x = [x for x, _ in bench_mark]
    y = [y for _, y in bench_mark]
    plt.plot(x, y, "bo")

    plt.show()


if __name__ == "__main__":
    main()
