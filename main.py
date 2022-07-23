from __future__ import annotations
import random
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from time import time

import src.Grid as Grid

Point = Grid.Point
Map = Grid.Map


def generate_bench_points(
    n: int, grid: Map, file_name: str = "points.txt"
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


def plot_result(
    grid: Map, result: list[Point], closed: list[Point], opened: list[Point]
) -> None:
    plot_grid = [
        [[y * 255] * 3 if type(y) == int else [255, 0, 0] for y in x] for x in grid
    ]
    origin = result[0]
    destiny = result[-1]
    for i in opened:
        if i not in result:
            plot_grid[i[1]][i[0]] = [122, 122, 0]
    for i in closed:
        if i not in result:
            plot_grid[i[1]][i[0]] = [255, 0, 0]
    plt.imshow(plot_grid)
    x = [x for x, _ in result]
    y = [y for _, y in result]
    plt.plot(x, y, "g-", lw=2)
    plt.plot(origin[0], origin[1], "bo")
    plt.plot(destiny[0], destiny[1], "go")
    plt.show()


def plot_side_by_side(
    grid: Map,
    origin: Point,
    destiny: Point,
    results: list[tuple[list[Point], list[Point], list[Point]], str],
    file_name: str = "result.png",
) -> None:
    _, axs = plt.subplots(1, len(results))
    for idx, res in enumerate(results):
        sub: Axes = axs[idx]
        result, closed, opened, title = res
        plot_grid = [
            [[y * 255] * 3 if type(y) == int else [255, 0, 0] for y in x] for x in grid
        ]
        for i in opened:
            plot_grid[i[1]][i[0]] = [122, 122, 0]
        for i in closed:
            plot_grid[i[1]][i[0]] = [255, 0, 0]
        for i in result:
            plot_grid[i[1]][i[0]] = [0, 255, 0]
        sub.set_title(title)
        sub.imshow(plot_grid)
        x = [x for x, _ in result]
        y = [y for _, y in result]
        sub.plot(x, y, "g-", lw=1.5)
        sub.plot(origin[0], origin[1], "bo")
        sub.plot(destiny[0], destiny[1], "go" if result else "ro")
    plt.savefig("figs/" + file_name, dpi=300)


def benchmark(
    points: list[Point],
    grid: Grid.Grid,
    tests_per_point: int = 2,
    plot: bool = False,
    file_name: str = "results.txt",
) -> None:
    visited_points: set[tuple[Point, Point]] = set()
    n: int = len(points) - 1
    limit = n * (n + 1) // 2
    with open(file_name, "w") as f:
        for idx, origin in enumerate(points):
            tests: int = 0
            print(f"\nProgress: {idx+1}/{n}")
            while tests < tests_per_point and len(visited_points) < limit:
                destiny: Point = None
                while not destiny:
                    next = random.choice(points)
                    if (
                        next != origin
                        and (next, origin) not in visited_points
                        and (origin, next) not in visited_points
                    ):
                        destiny = next
                result_str = f"{origin} {destiny} "

                start_time = time()
                print(f"Origin: {origin} - Destination: {destiny}")
                result, weight, closed, opened = grid.a_grid_graph_search(
                    origin, destiny
                )
                end_time = time() - start_time
                print(f"A* with Visibility Graph finished in {end_time} seconds")
                print("Weight: ", weight)
                results = [(result, closed, opened, "A* with Visibility Graph")]
                result_str += f"{weight},{len(closed)},{end_time},"

                start_time = time()
                print("-- A* without Visibility Graph --")
                result, weight, closed, opened = grid.a_grid_search(origin, destiny)
                end_time = time() - start_time
                print(f"A* finished in {end_time} seconds")
                print("Weight: ", weight)
                results.append((result, closed, opened, "A* in Raw Grid"))
                result_str += f"{weight},{len(closed)},{end_time}\n"

                if plot:
                    plot_side_by_side(
                        grid.grid, origin, destiny, results, f"{origin},{destiny}.png"
                    )
                visited_points.add((origin, destiny))
                tests += 1
                f.write(result_str)


def main():

    grid = [
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]

    tsg = Grid.TSG(grid)
    tsg.create_from_file("maps/Berlin_0_1024.map")
    bench_points = read_bench_points()
    benchmark(bench_points, tsg, plot=True)


if __name__ == "__main__":
    main()
