from __future__ import annotations
import random
import matplotlib.pyplot as plt
from time import time

import src.Grid as Grid

Point = Grid.Point


def generate_bench_points(
    n: int, grid: list[list], file_name: str = "points.txt"
) -> list[Point]:
    bench_points: list[Point] = []
    max_tries = n * n
    with open(file_name, "w") as f:
        while max_tries and len(bench_points) < n:
            max_tries -= 1
            x = int(len(grid[0]) * random.random())
            y = int(len(grid) * random.random())
            if (x, y) in bench_points:
                continue
            if grid[y][x] != 0:
                bench_points.append((x, y))
                f.write(f"{x} {y}\n")
    return bench_points


def read_bench_points(file_name: str = "points.txt") -> list[Point]:
    bench_points: list[Point] = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            line = line.split()
            x = int(line[0])
            y = int(line[1])
            bench_points.append((x, y))
    return bench_points


def benchmark(
    points: list[Point], grid: Grid.Grid, tests_per_point: int = 2, plot: bool = True
):
    visited_points: set[tuple[Point, Point]] = set()
    n: int = len(points) - 1
    limit = n * (n + 1) // 2
    for p in points:
        tests = 0
        while tests < tests_per_point and len(visited_points) < limit:
            destiny: Point = None
            while not destiny:
                next = random.choice(points)
                if (
                    next != p
                    and (next, p) not in visited_points
                    and (p, next) not in visited_points
                ):
                    destiny = next
            start_time = time()
            result, weight, close, open = grid.a_grid_graph_search(p, destiny)
            print(f"Origin: {p} - Destination: {destiny}")
            print(f"A* with Visibility Graph finished in {time() - start_time} seconds")
            print("Weight: ", weight)
            visited_points.add((p, destiny))
            tests += 1
            if plot:
                plot_grid = [
                    [[y * 255] * 3 if type(y) == int else [255, 0, 0] for y in x]
                    for x in grid.grid
                ]
                for i in open:
                    if i not in result:
                        plot_grid[i[1]][i[0]] = [122, 122, 0]
                for i in close:
                    if i not in result:
                        plot_grid[i[1]][i[0]] = [255, 0, 0]
                plt.imshow(plot_grid)
                x = [x for x, _ in result]
                y = [y for _, y in result]
                plt.plot(x, y, "g-", lw=2)
                plt.plot(p[0], p[1], "bo")
                plt.plot(destiny[0], destiny[1], "go")
                plt.show()


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
    tsg.create_from_file("maps/Berlin_0_1024.map")

    bench_points = read_bench_points()
    benchmark(bench_points, tsg)


if __name__ == "__main__":
    main()
