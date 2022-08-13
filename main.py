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
    """
    Function to generate a list of n random valid points on the grid. The points are
    returned as a list of tuples and saved in a file.
    """
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
    """
    Reads a file and returns a list of points.
    """
    bench_points: list[Point] = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            line = line.split()
            x = int(line[0])
            y = int(line[1])
            bench_points.append((x, y))
    return bench_points


def plot_result(
    grid: Map,
    origin: Point,
    destiny: Point,
    result: list[Point],
    closed: list[Point],
    opened: list[Point],
) -> None:
    """
    Plots the result of the search with opened, closed and result nodes. The result is
    displayed in the window.
    """
    plot_grid = [
        [[y * 255] * 3 if type(y) == int else [255, 0, 0] for y in x] for x in grid
    ]
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
    plt.clf()
    plt.cla()
    plt.close()


def plot_side_by_side(
    grid: Map,
    origin: Point,
    destiny: Point,
    results: list[tuple[list[Point], list[Point], list[Point]], str],
    file_name: str = "result.png",
) -> None:
    """
    Plot N resuls of the search, displayed side by side to compare. The results are
    stored in a file or displayed in the window.
    """
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
    if file_name:
        plt.savefig("figs/" + file_name, dpi=300)
    else:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_resume(file: str = "results.txt") -> None:
    """
    Plot and compare the results of two algorithms.
    """

    with open(file, "r") as txt:
        # Referencing graph
        equal_w = greater_w = lower_w = 0
        graph_wheights = []
        grid_wheights = []
        graph_times = []
        grid_times = []
        graph_nodes = []
        grid_nodes = []
        same_w: list[tuple] = []
        for line in txt.readlines():
            line = line.strip().split(";")
            (
                origin,
                destiny,
                graph_w,
                graph_node,
                graph_time,
                grid_w,
                grid_node,
                grid_time,
            ) = line
            graph_w = int(graph_w)
            grid_w = int(grid_w)
            graph_wheights.append(graph_w)
            grid_wheights.append(grid_w)
            graph_times.append(float(graph_time))
            grid_times.append(float(grid_time))
            graph_nodes.append(int(graph_node))
            grid_nodes.append(int(grid_node))
            equal_w += graph_w == grid_w
            lower_w += graph_w < grid_w
            greater_w += graph_w > grid_w
            if graph_w == grid_w:
                same_w.append(
                    (origin, destiny, graph_w, float(graph_time), float(grid_time))
                )
        graph_t = [x[3] for x in same_w]
        grid_t = [x[4] for x in same_w]
        worst_graph = max(same_w, key=lambda x: x[3])
        worst_grid = max(same_w, key=lambda x: x[4])
        best_graph = min(same_w, key=lambda x: x[3])
        best_grid = min(same_w, key=lambda x: x[4])
        print(f"Same weight results: {len(same_w)}")
        print(f"Avarage time with visibility graph: {sum(graph_t) / len(same_w)}")
        print(f"Avarage time without visibility graph: {sum(grid_t) / len(same_w)}")
        print(
            f"Worst time with visibility graph: {worst_graph[3]}s with a cost "
            + f"of {worst_graph[2]} from {worst_graph[0]} to {worst_graph[1]}"
        )
        print(
            f"Worst time without visibility graph: {worst_grid[4]}s with a cost "
            + f"of {worst_grid[2]} from {worst_grid[0]} to {worst_grid[1]}"
        )
        print(
            f"Best time with visibility graph: {best_graph[3]}s with a cost "
            + f"of {best_graph[2]} from {best_graph[0]} to {best_graph[1]}"
        )
        print(
            f"Best time without visibility graph: {best_grid[4]}s with a cost "
            + f"of {best_grid[2]} from {best_grid[0]} to {best_grid[1]}"
        )

        data = [lower_w, equal_w, greater_w]

        _, axs = plt.subplots(2, 2)
        axs[0][0].set_title("Weight comparison")
        axs[0][0].bar(["Graph < Grid", "Equal", "Graph > Grid"], data)

        graw = sorted(graph_wheights)
        griw = sorted(grid_wheights)
        axs[1][0].set_title("Weight distribution")
        axs[1][0].scatter(
            [x for x in range(len(griw))],
            graw,
            marker="o",
            label="With Visibility Graph",
            s=20.0,
            c="b",
        )
        axs[1][0].scatter(
            [x for x in range(len(graw))],
            griw,
            marker="o",
            label="Without Visibility Graph",
            s=10.0,
            c="r",
        )
        axs[1][0].legend(
            loc="upper left", markerscale=2.0, scatterpoints=1, fontsize=10
        )
        axs[1][0].set_ylabel("Weight")
        axs[1][0].axes.get_xaxis().set_visible(False)

        axs[0][1].set_title("Time comparison")
        axs[0][1].set_ylabel("Time (seconds)")
        axs[0][1].set_xlabel("Weight")
        axs[0][1].scatter(
            graph_wheights,
            graph_times,
            marker="o",
            label="With Visibility Graph",
            s=20.0,
            c="b",
        )
        axs[0][1].scatter(
            grid_wheights,
            grid_times,
            marker="o",
            label="Without Visibility Graph",
            s=10.0,
            c="r",
        )
        axs[0][1].legend(loc="best", markerscale=2.0, scatterpoints=1, fontsize=10)

        axs[1][1].set_title("Nodes comparison")
        axs[1][1].set_ylabel("Visited nodes")
        axs[1][1].set_xlabel("Weight")
        axs[1][1].scatter(
            graph_wheights,
            graph_nodes,
            marker="o",
            label="With Visibility Graph",
            s=20.0,
            c="b",
        )
        axs[1][1].scatter(
            grid_wheights,
            grid_nodes,
            marker="o",
            label="Without Visibility Graph",
            s=10.0,
            c="r",
        )
        axs[1][1].legend(loc="best", markerscale=2.0, scatterpoints=1, fontsize=10)

        plt.show()
        plt.clf()
        plt.cla()
        plt.close()


def benchmark(
    points: list[Point],
    grid: Grid.Grid,
    tests_per_point: int = 2,
    plot: bool = False,
    file_name: str = "results.txt",
) -> None:
    """
    Given a list of valid points and a grid, the function runs the search for each
    point to store the results into a file and plot it into a image. Two points are
    not visited twice and the results are used to compare the performance between a
    simple A* search and with the help of visualization graph.
    """
    visited_points: set[tuple[Point, Point]] = set()
    n: int = len(points) - 1
    limit = n * (n + 1) // 2
    with open(file_name, "w") as f:
        for idx, origin in enumerate(points):
            tests: int = 0
            print(f"\nProgress: {idx}/{n}")
            while tests < tests_per_point and len(visited_points) < limit:
                destiny: Point = None
                # Get a random destiny point from the list of points that was not visited by origin
                while not destiny:
                    next = random.choice(points)
                    if (
                        next != origin
                        and (next, origin) not in visited_points
                        and (origin, next) not in visited_points
                        and Grid.Grid.h_distance(origin, next) > 1000
                    ):
                        destiny = next
                result_str = f"{origin};{destiny};"

                # A* search with Visibility Graph
                start_time = time()
                print(f"Origin: {origin} - Destination: {destiny}")
                result, weight, closed, opened = grid.a_grid_graph_search(
                    origin, destiny
                )
                end_time = time() - start_time
                print(f"A* with Visibility Graph finished in {end_time} seconds")
                print("Weight: ", weight)
                results = [(result, closed, opened, "A* with Visibility Graph")]
                result_str += f"{weight};{len(closed)};{end_time};"

                # A* search without Visibility Graph
                start_time = time()
                print("-- A* without Visibility Graph --")
                result, weight, closed, opened = grid.a_grid_search(origin, destiny)
                end_time = time() - start_time
                print(f"A* finished in {end_time} seconds")
                print("Weight: ", weight)
                results.append((result, closed, opened, "A* in Raw Grid"))
                result_str += f"{weight};{len(closed)};{end_time}\n"

                if plot:
                    plot_side_by_side(
                        grid.grid, origin, destiny, results, f"{origin},{destiny}.png"
                    )
                visited_points.add((origin, destiny))
                tests += 1
                f.write(result_str)


def main():
    tsg = Grid.TSG()
    tsg.create_from_file("maps/Milan_0_1024.map")
    bench_points = generate_bench_points(100, tsg.grid)
    benchmark(bench_points, tsg, plot=True)
    plot_resume()


if __name__ == "__main__":
    main()
