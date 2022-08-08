from __future__ import annotations

from matplotlib import pyplot as plt
from .Vertex import Vertex

from time import time
import numpy as np
import cv2 as cv

# Travel weights
DIAGONAL = 14
NORMAL = 10
# Minimal grid size to use the Corner's algorithm
MIN_SIZE = 15
# If true, the simple vertice detection method is used. If false, the Corner Harris one is used
USE_OPENCV = True
# Limit of vertices alowed to be around another one
MAX_VERTICES_PER_VERTEX = 1

Point = tuple[int, int]
Map = list[list]

# List of movements allowed to be around a vertex
movements: tuple[int, int] = [
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
]


class Grid:
    def __init__(self, grid: Map = [[]]) -> None:
        self.grid = grid

    def __str__(self) -> str:
        return str(self.grid)

    def _is_out_of_bounds(self, x: int, y: int) -> bool:
        """
        Funcion to check if given coordinates are outside the grid
        """
        return x < 0 or y < 0 or y >= len(self.grid) or x >= len(self.grid[y])

    def _is_corner(self, cell: Point, dir_x: int, dir_y: int):
        """
        Function to check if a Point is a corner related to a direction
        """
        x = cell[0]
        y = cell[1]
        pos = (x + dir_x, y + dir_y)
        if self._is_out_of_bounds(x, y) or self._is_out_of_bounds(pos[0], pos[1]):
            return False
        return (
            self.grid[pos[1]][pos[0]] == 0
            and self.grid[y][pos[0]] != 0 != self.grid[pos[1]][x]
        )

    def plot_grid(self) -> None:
        """
        Plot the grid
        """
        plot_grid = [
            [[y * 255] * 3 if type(y) == int else [255, 0, 0] for y in x]
            for x in self.grid
        ]
        plt.imshow(plot_grid)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def a_grid_graph_search(
        self, origin: Point, destiny: Point
    ) -> tuple[list[Point], int, list[Point]]:
        """
        Interface to A* search using visibility graph
        """
        return [], -1, []

    def a_graph_search(
        self,
        origin: Point,
        destiny: Point,
        skip: set[str] = [],
        limit: int = -1,
    ) -> tuple[list[Vertex], int]:
        """
        Interface to A* search into visibility graph
        """
        return [], -1

    def a_grid_search(
        self, origin: Point, destiny: Point, limit: int = -1
    ) -> tuple[list[Point], int, list[Point], list[Point]]:
        """
        A* search algorithm given an origin and the destiny
        """

        class Cell:
            """
            Auxiliar class to store opened and closed nodes and their weights
            """

            def __init__(self, x: int, y: int, weight: int, father: "Cell") -> None:
                self.x = x
                self.y = y
                self.father = father
                self.weight = weight
                self.heuristic = self.weight + Grid.h_distance((x, y), destiny)

            def has_diagonal(
                self,
                dir_x: int,
                dir_y: int,
                grid: list[list],
            ) -> bool:
                """
                Function to check if the current cell has access to its diagonal given it
                his direction
                """
                return Vertex.has_diagonal((self.x, self.y), dir_x, dir_y, grid)

            def is_destiny(self, target: Point) -> bool:
                return self.x == target[0] and self.y == target[1]

            def __lt__(self, other: "Cell") -> bool:
                return self.heuristic < other.heuristic

        key = str(origin[0]) + "," + str(origin[1])
        opened_nodes: dict[str, Cell] = {key: Cell(origin[0], origin[1], 0, None)}
        closed_nodes: dict[str, Cell] = {}
        while opened_nodes:
            # Get the node with the lowest heuristic
            lowest = min(opened_nodes.values())
            key = str(lowest.x) + "," + str(lowest.y)
            curr = opened_nodes.pop(key)
            if curr.is_destiny(destiny):
                weight = curr.weight
                path: list[Point] = []
                # Get the reverse path from the destiny to the origin
                while curr != None:
                    path.append((curr.x, curr.y))
                    curr = curr.father
                path.reverse()
                return (
                    path,
                    weight,
                    [(closed_nodes[i].x, closed_nodes[i].y) for i in closed_nodes],
                    [(opened_nodes[i].x, opened_nodes[i].y) for i in opened_nodes],
                )
            # If the lowest node overcame the limit, return empty path
            if limit > 0 and curr.heuristic > limit:
                return [], -1, [], []
            # Closes the node
            closed_nodes[key] = curr
            to_x = curr.x - destiny[0]
            to_y = curr.y - destiny[1]
            direct = abs(to_x) > abs(to_y)
            way = to_x > 0 if direct else to_y > 0
            # Prioritizes the movements towards the destiny
            sorted_movements = sorted(movements, key=lambda x: x[direct], reverse=way)
            for move in sorted_movements:
                m_x = move[0] + curr.x
                m_y = move[1] + curr.y
                next_key = str(m_x) + "," + str(m_y)
                is_diagonal = move[0] != 0 and move[1] != 0
                next_weight = curr.weight + (DIAGONAL if is_diagonal else NORMAL)
                # Check if the node is valid or has been already visited
                if (
                    next_key in closed_nodes
                    or self._is_out_of_bounds(m_x, m_y)
                    or self.grid[m_y][m_x] == 0
                    or (is_diagonal and not curr.has_diagonal(*move, self.grid))
                ):
                    continue
                distance = Grid.h_distance((m_x, m_y), destiny)
                if limit > 0 and distance + next_weight > limit:
                    continue
                # Create the node or update the heuristic if it has been visited and has a lower one
                if next_key in opened_nodes:
                    if next_weight + distance < opened_nodes[next_key].heuristic:
                        opened_nodes[next_key].weight = next_weight
                        opened_nodes[next_key].heuristic = next_weight + distance
                        opened_nodes[next_key].father = curr
                else:
                    opened_nodes[next_key] = Cell(m_x, m_y, next_weight, curr)

        return (
            [],
            -1,
            [(closed_nodes[i].x, closed_nodes[i].y) for i in closed_nodes],
            [(opened_nodes[i].x, opened_nodes[i].y) for i in opened_nodes],
        )

    def read_file(self, file_name: str) -> None:
        """
        Create the grid from a .map file
        """
        start = time()
        self.grid = [[]]
        with open(file_name, "r") as f:
            f.readline()  # figure type
            h_size = int(f.readline().strip().split(" ")[1])  # height
            f.readline()  # width
            f.readline()  # map
            self.grid = [[]] * h_size
            for index, row in enumerate(f.readlines()):
                self.grid[index] = [1 if x == "." else 0 for x in row.strip()]
        print(
            f"{len(self.grid[0])}x{len(self.grid)} grid readed in {time() - start} seconds"
        )

    def h_distance(origin: Point, destiny: Point) -> int:
        """
        Returns the octile distance between two coordinates pairs
        """
        dist_x = abs(origin[0] - destiny[0])
        dist_y = abs(origin[1] - destiny[1])
        return NORMAL * max(dist_x, dist_y) + (DIAGONAL - NORMAL) * min(dist_x, dist_y)


class SSG(Grid):
    """
    Simple SubLevel Graph
    """

    def __init__(self, grid: Map = [[]]) -> None:
        super().__init__(grid)
        self.vertices: dict[str, Vertex] = {}

    def plot_edges(self) -> None:
        """
        Plot the grid
        """
        plot_grid = [
            [[y * 255] * 3 if type(y) == int else [255, 0, 0] for y in x]
            for x in self.grid
        ]
        edges_x = [
            (y.origin.x, y.destiny.x)
            for x in self.vertices.values()
            for y in x.edges.values()
        ]
        edges_x = [[x[0] for x in edges_x], [x[1] for x in edges_x]]
        edges_y = [
            (y.origin.y, y.destiny.y)
            for x in self.vertices.values()
            for y in x.edges.values()
        ]
        edges_y = [[y[0] for y in edges_y], [y[1] for y in edges_y]]

        plt.imshow(plot_grid)
        plt.plot(edges_x, edges_y, "b-", lw=0.3)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def create_graph(self) -> None:
        """
        All steps needed to create visibility graph
        """
        self.create_vertices()
        # self.reduce_vertices()
        self.create_edges()
        self.reduce_edges()

    def create_edges(self) -> None:
        start = time()
        for vertex in self.vertices:
            self.vertices[vertex].create_edges()
        print(f"Edges created in {time() - start} seconds")

    def reduce_edges(self) -> None:
        start = time()
        for vertex in self.vertices:
            self.vertices[vertex].reduce_edges()
        print(f"Edges reduced in {time() - start} seconds")

    def create_from_file(self, file_name: str) -> None:
        self.read_file(file_name)
        self.create_graph()

    def a_grid_graph_search(
        self, origin: Point, destiny: Point
    ) -> tuple[list[Point], int, list[Point]]:
        """
        A* search between two points using the visibility graph help
        """
        print("-- A* Grid Graph Search --")
        if (
            self._is_out_of_bounds(origin[0], origin[1])
            or self._is_out_of_bounds(destiny[0], destiny[1])
            or self.grid[origin[1]][origin[0]] == 0
            or self.grid[destiny[1]][destiny[0]] == 0
        ):
            return [], -1, []
        start_is_vertex = False
        end_is_vertex = False
        # Creates the origin and destiny as vertices from the visibility graph
        if self.grid[origin[1]][origin[0]] == 1:
            vertex = Vertex(origin[0], origin[1], self.grid)
            s_key = str(origin[0]) + "," + str(origin[1])
            self.grid[origin[1]][origin[0]] = vertex
            self.vertices[s_key] = vertex
            start_is_vertex = True
        if self.grid[destiny[1]][destiny[0]] == 1:
            vertex = Vertex(destiny[0], destiny[1], self.grid)
            e_key = str(destiny[0]) + "," + str(destiny[1])
            self.grid[destiny[1]][destiny[0]] = vertex
            self.vertices[e_key] = vertex
            end_is_vertex = True
        start: Vertex = self.grid[origin[1]][origin[0]]
        end: Vertex = self.grid[destiny[1]][destiny[0]]
        start.create_edges()
        start.reduce_edges()
        end.create_edges()
        end.reduce_edges()

        # Find shortest vertices path
        path, w = self.a_graph_search((start.x, start.y), (end.x, end.y))

        # If origin or destiny are not vertices, remove them from the grid
        if start_is_vertex:
            edge_keys = [str(e) for e in start.edges]
            for e in edge_keys:
                edge = start.edges[e]
                start.del_edge(edge.destiny.x, edge.destiny.y)
            self.grid[origin[1]][origin[0]] = 1
            self.vertices.pop(s_key)
        if end_is_vertex:
            edge_keys = [str(e) for e in end.edges]
            for e in edge_keys:
                edge = end.edges[e]
                end.del_edge(edge.destiny.x, edge.destiny.y)
            self.grid[destiny[1]][destiny[0]] = 1
            self.vertices.pop(e_key)

        if not path:
            return [], -1, []

        # Converts the vertices path into a grid path, using a simple A* search between each node
        grid_weight = 0 if path else -1
        vertex = path.pop(0)
        grid_path: list[Point] = []
        closed_nodes: list[Point] = []
        opened_nodes: list[Point] = []
        while path:
            next = path.pop(0)
            p, w, c, o = self.a_grid_search((vertex.x, vertex.y), (next.x, next.y))
            grid_weight += w
            grid_path.extend(p)
            closed_nodes.extend(c)
            opened_nodes.extend(o)
            vertex = next

        return grid_path, grid_weight, closed_nodes, opened_nodes

    def __create_simple_vertice(self):
        """
        Funcion to iterate through and detect if a pixel can be a vertex
        """

        self.vertices: dict[str, Vertex] = {}
        for y, row in enumerate(self.grid):
            for x, value in enumerate(row):
                if value == 0:
                    continue
                key = str(x) + "," + str(y)
                cell = (x, y)
                corners: list[tuple[int, int]] = [
                    (-1, -1),  # Superior left
                    (1, -1),  # Superior right
                    (1, 1),  # Inferior right
                    (-1, 1),  # Inferior left
                ]
                for corner in corners:
                    d_x = corner[0]
                    d_y = corner[1]
                    if self._is_corner(cell, d_x, d_y):
                        vertex = Vertex(x, y, self.grid)
                        self.vertices[key] = vertex
                        self.grid[y][x] = vertex
                        break
                else:
                    self.grid[y][x] = 1

    def create_vertices(self) -> None:
        """
        Function that detects vertices in the grid using opencv Corner Harris Detection
        Avaliable at: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
        """

        start = time()
        self.vertices: dict[str, Vertex] = {}

        if not USE_OPENCV or len(self.grid) < MIN_SIZE or len(self.grid[0]) < MIN_SIZE:
            self.__create_simple_vertice()
            print(f"{len(self.vertices)} vertices created in {time() - start} seconds")
            return

        plot_grid = np.float32(
            [[[y * 255, y * 255, y * 255] for y in x] for x in self.grid]
        )
        gray = cv.cvtColor(plot_grid, cv.COLOR_BGR2GRAY)
        # find Harris corners
        gray = np.float32(gray)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = cv.dilate(dst, None)
        new_plt = dst > 0.01 * dst.max()
        indexes = np.where(new_plt == True)
        for x, y in zip(indexes[1], indexes[0]):
            if self.grid[y][x] == 0:
                continue
            corners: list[tuple[int, int]] = [
                (-1, -1),  # Superior left
                (1, -1),  # Superior right
                (1, 1),  # Inferior right
                (-1, 1),  # Inferior left
            ]
            for corner in corners:
                d_x = corner[0]
                d_y = corner[1]
                if self._is_corner((x, y), d_x, d_y):
                    vertex = Vertex(x, y, self.grid)
                    self.grid[y][x] = vertex
                    self.vertices[vertex.key] = vertex
                    break
            else:
                self.grid[y][x] = 1
        print(f"{len(self.vertices)} vertices created in {time() - start} seconds")

    def reduce_vertices(self) -> None:
        """
        Kills a vertex if it has more than MAX_VERTICES_PER_VERTEX neighbors
        """
        start = time()
        to_delete: list[str] = []
        for goal_key in self.vertices:
            if goal_key in to_delete:
                continue
            vertex = self.vertices[goal_key]
            count_neighbors = 0
            for move in movements:
                key = str(move[0] + vertex.x) + "," + str(move[1] + vertex.y)
                if key in self.vertices:
                    count_neighbors += 1
            if count_neighbors > MAX_VERTICES_PER_VERTEX:
                self.grid[vertex.y][vertex.x] = 1
                to_delete.append(goal_key)
        self.vertices = dict(
            [(x, self.vertices[x]) for x in self.vertices if x not in to_delete]
        )
        print(f"{len(to_delete)} vertices reduced in {time() - start} seconds")

    def a_graph_search(
        self,
        origin: Point,
        destiny: Point,
        skip: set[str] = [],
        limit: int = -1,
    ) -> tuple[list[Vertex], int]:
        """
        A* search algorithm between two valid vertices
        """

        class Cell:
            """
            Auxiliar class to store opened and closed nodes and their weights
            """

            def __init__(self, vertex: Vertex, weight: int, father: "Cell") -> None:
                self.vertex = vertex
                self.father = father
                self.weight = weight
                self.heuristic = self.weight + Grid.h_distance(
                    (vertex.x, vertex.y), destiny
                )

            def is_destiny(self, target: Point) -> bool:
                return self.vertex.x == target[0] and self.vertex.y == target[1]

            def __lt__(self, other: "Cell") -> bool:
                return self.heuristic < other.heuristic

        key = str(origin[0]) + "," + str(origin[1])
        opened_nodes: dict[str, Cell] = {key: Cell(self.vertices[key], 0, None)}
        closed_nodes: dict[str, Cell] = {}
        while opened_nodes:
            # Get the node with the lowest heuristic
            lowest = min(opened_nodes.values())
            key = lowest.vertex.key
            curr = opened_nodes.pop(key)
            if curr.is_destiny(destiny):
                weight = curr.weight
                path: list[Vertex] = []
                while curr != None:
                    path.append(curr.vertex)
                    curr = curr.father
                path.reverse()
                return path, weight
            # If the lowest node overcame the limit, return empty path
            if limit > 0 and curr.heuristic > limit:
                return [], -1
            # Closes the node
            closed_nodes[key] = curr
            curr_vertex = curr.vertex
            for edge in curr_vertex.edges:
                curr_edge = curr_vertex.edges[edge]
                next = curr_edge.destiny
                next_key = next.key
                # Check if the node is valid or has been already visited
                if next_key in closed_nodes or next_key in skip:
                    continue
                next_weight = curr.weight + curr_edge.weight
                distance = Grid.h_distance((next.x, next.y), destiny)
                # Create the node or update the heuristic if it has been visited and has a lower one
                if next_key in opened_nodes:
                    if next_weight + distance < opened_nodes[next_key].heuristic:
                        opened_nodes[next_key].heuristic = next_weight + distance
                        opened_nodes[next_key].father = curr
                else:
                    opened_nodes[next_key] = Cell(next, next_weight, curr)

        return [], -1

    def h_reachable(self, origin: "Vertex", destiny: Point) -> bool:
        """
        Function to check if there is a shortest path between two points whose length is equal
        to the heuristic between them (they are h-reachable)
        """

        def clearence(
            origin: Point,
            dir_x: int,
            dir_y: int,
        ) -> int:
            curr = (origin[0] + dir_x, origin[1] + dir_y)
            max: int = 0
            while not self._is_out_of_bounds(curr[0] + dir_x, curr[1] + dir_y):
                if self.grid[curr[1]][curr[0]] == 0:
                    return max
                if curr == destiny:
                    return max + 1
                curr = (curr[0] + dir_x, curr[1] + dir_y)
                max += 1
            return max

        dir_x = min(max(-1, destiny[0] - origin.x), 1)
        dir_y = min(max(-1, destiny[1] - origin.y), 1)
        curr = (origin.x, origin.y)
        while Vertex.has_diagonal(curr, dir_x, dir_y, self.grid):
            curr = (curr[0] + dir_x, curr[1] + dir_y)
            lim_h = clearence(curr, dir_x, 0) if dir_x else 0
            lim_v = clearence(curr, 0, dir_y) if dir_y else 0
            if curr[1] == destiny[1] and curr[0] + (dir_x * lim_h) == destiny[0]:
                return True
            if curr[1] + (dir_y * lim_v) == destiny[1] and curr[0] == destiny[0]:
                return True
            if curr[1] == destiny[1] or curr[0] == destiny[0]:
                return False
        return False


class TSG(SSG):
    """
    Two SubLevel Graph
    """

    def __init__(self, grid: Map = [[]]) -> None:
        super().__init__(grid)
        self.local_goals: dict[str, Vertex] = {}

    def plot_edges(self) -> None:
        """
        Plot the grid
        """
        plot_grid = [
            [[y * 255] * 3 if type(y) == int else [255, 0, 0] for y in x]
            for x in self.grid
        ]
        edges_x = [
            (y.origin.x, y.destiny.x)
            for x in self.vertices.values()
            for y in x.edges.values()
            if not y.is_local
        ]
        edges_x = [[x[0] for x in edges_x], [x[1] for x in edges_x]]
        edges_y = [
            (y.origin.y, y.destiny.y)
            for x in self.vertices.values()
            for y in x.edges.values()
            if not y.is_local
        ]
        edges_y = [[y[0] for y in edges_y], [y[1] for y in edges_y]]

        plt.imshow(plot_grid)
        plt.plot(edges_x, edges_y, "b-", lw=0.3)
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def create_from_file(self, file_name: str) -> None:
        self.read_file(file_name)
        self.create_graph()

    def create_graph(self) -> None:
        super().create_graph()
        self.convert_to_tsg()

    def convert_to_tsg(self) -> None:
        """
        Converts the Simple SubLevel Graph to a Two SubLevel Graph by identifying the
        vertices that can be classified as local goals
        """
        start = time()
        global_goals: dict[str, Vertex] = dict(self.vertices)
        for s in self.vertices:
            vert = self.vertices[s]
            new_edges: dict[str, tuple[Vertex, Vertex]] = {}
            is_local: bool = True
            visited_edges: set[tuple[str, str]] = set()
            # Iterate over all neighbors of the vertex to identify local goals
            for p in vert.edges:
                for q in vert.edges:
                    if p == q or (q, p) in visited_edges:
                        continue
                    visited_edges.add((p, q))
                    skip = set([x for x in self.local_goals if x != p and x != q])
                    skip.add(s)
                    origin = vert.edges[p].destiny
                    destiny = vert.edges[q].destiny
                    limit = vert.edges[p].weight + vert.edges[q].weight
                    _, d = self.a_graph_search(
                        (origin.x, origin.y), (destiny.x, destiny.y), skip, limit
                    )
                    # If when removing the vertex, the distance between the neighbors
                    # that not passes between another local goal increases and they are
                    # not h-reachable, then the vertex is a global goal
                    if d == -1 or d > limit:
                        if self.h_reachable(origin, (destiny.x, destiny.y)):
                            check_key = q + "," + p
                            if check_key not in new_edges:
                                new_edges[p + "," + q] = (origin, destiny)
                        else:
                            is_local = False
                            break
                if not is_local:
                    break
            # Creates the necessary edges to connect the local goals
            if is_local:
                for edge in new_edges:
                    edge_tuple = new_edges[edge]
                    edge_tuple[0].add_edge(edge_tuple[1])
                self.local_goals[s] = vert

        # Removes the local goals from the list and stores the global goals into "vertices" variable
        for l in self.local_goals:
            goal = self.local_goals[l]
            for e in goal.edges:
                edge = goal.edges[e]
                edge.is_local = True
                edge.destiny.edges[goal.key].is_local = True
        self.vertices = dict(
            [(x, global_goals[x]) for x in global_goals if x not in self.local_goals]
        )
        print(
            f"TSG converted in {time() - start} seconds, {len(self.local_goals)} local goals detected"
        )

    def a_grid_graph_search(
        self, origin: Point, destiny: Point
    ) -> tuple[list[Point], int, list[Point], list[Point]]:
        """
        A* search between two points using the visibility graph help
        """
        print("-- A* Grid Graph Search --")
        if (
            self._is_out_of_bounds(origin[0], origin[1])
            or self._is_out_of_bounds(destiny[0], destiny[1])
            or self.grid[origin[1]][origin[0]] == 0
            or self.grid[destiny[1]][destiny[0]] == 0
        ):
            return [], -1, [], []
        start_is_vertex = False
        end_is_vertex = False
        # Creates the origin and destiny as vertices from the visibility graph
        if self.grid[origin[1]][origin[0]] == 1:
            vertex = Vertex(origin[0], origin[1], self.grid)
            s_key = str(origin[0]) + "," + str(origin[1])
            self.grid[origin[1]][origin[0]] = vertex
            self.vertices[s_key] = vertex
            start_is_vertex = True
        if self.grid[destiny[1]][destiny[0]] == 1:
            vertex = Vertex(destiny[0], destiny[1], self.grid)
            e_key = str(destiny[0]) + "," + str(destiny[1])
            self.grid[destiny[1]][destiny[0]] = vertex
            self.vertices[e_key] = vertex
            end_is_vertex = True
        start: Vertex = self.grid[origin[1]][origin[0]]
        end: Vertex = self.grid[destiny[1]][destiny[0]]
        start.create_edges()
        start.reduce_edges()
        end.create_edges()
        end.reduce_edges()

        # Temporarily adds the direct origin and destiny local goals to the list of vertices
        for e in start.edges:
            edge_keys = [str(e) for e in start.edges]
            for e in edge_keys:
                d = start.edges[e].destiny
                if d.key in self.local_goals:
                    self.vertices[d.key] = self.local_goals.pop(d.key)
        for e in end.edges:
            edge_keys = [str(e) for e in end.edges]
            for e in edge_keys:
                d = end.edges[e].destiny
                if d.key in self.local_goals:
                    self.vertices[d.key] = self.local_goals.pop(d.key)

        skip = set([x for x in self.local_goals])
        # Find shortest vertices path that not passes between any local goal
        path, w = self.a_graph_search((start.x, start.y), (end.x, end.y), skip=skip)

        # Removes the direct origin and destiny local goals from the list of vertices
        for e in start.edges:
            edge_keys = [str(e) for e in start.edges]
            for e in edge_keys:
                edge = start.edges[e]
                d = start.edges[e].destiny
                if edge.is_local:
                    self.local_goals[d.key] = self.vertices.pop(d.key)
        for e in end.edges:
            edge_keys = [str(e) for e in end.edges]
            for e in edge_keys:
                edge = end.edges[e]
                d = end.edges[e].destiny
                if edge.is_local:
                    self.local_goals[d.key] = self.vertices.pop(d.key)

        # If origin or destiny are not vertices, remove them from the grid
        if start_is_vertex:
            edge_keys = [str(e) for e in start.edges]
            for e in edge_keys:
                edge = start.edges[e]
                start.del_edge(edge.destiny.x, edge.destiny.y)
            self.grid[origin[1]][origin[0]] = 1
            self.vertices.pop(s_key)
        if end_is_vertex:
            edge_keys = [str(e) for e in end.edges]
            for e in edge_keys:
                edge = end.edges[e]
                end.del_edge(edge.destiny.x, edge.destiny.y)
            self.grid[destiny[1]][destiny[0]] = 1
            self.vertices.pop(e_key)

        if not path:
            return [], -1, [], []

        # Converts the vertices path into a grid path, using a simple A* search between each node
        grid_weight = 0
        vertex = path.pop(0)
        grid_path: list[Point] = []
        closed_nodes: list[Point] = []
        opened_nodes: list[Point] = []
        while path:
            next = path.pop(0)
            p, w, c, o = self.a_grid_search((vertex.x, vertex.y), (next.x, next.y))
            grid_weight += w
            grid_path.extend(p)
            closed_nodes.extend(c)
            opened_nodes.extend(o)
            vertex = next

        return grid_path, grid_weight, closed_nodes, opened_nodes
