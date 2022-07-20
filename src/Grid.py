from .Vertex import Vertex

SQRT_2 = 1.4


class Grid:
    def __init__(self, grid: list[list] = [[]]) -> None:
        self.grid = grid

    def __str__(self) -> str:
        return str(self.grid)

    def a_grid_search(
        self, origin: tuple[int, int], destiny: tuple[int, int]
    ) -> tuple[list, int]:
        class Cell:
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
                return Vertex.has_diagonal((self.x, self.y), dir_x, dir_y, grid)

            def is_destiny(self, target: tuple[int, int]) -> bool:
                return self.x == target[0] and self.y == target[1]

            def __lt__(self, other: "Cell") -> bool:
                return self.heuristic < other.heuristic

        key = str(origin[0]) + "," + str(origin[1])
        opened_nodes: dict[str, Cell] = {key: Cell(origin[0], origin[1], 0, None)}
        closed_nodes: dict[str, Cell] = {}
        while opened_nodes:
            lowest = min(opened_nodes.values())
            key = str(lowest.x) + "," + str(lowest.y)
            curr = opened_nodes.pop(key)
            if curr.is_destiny(destiny):
                weight = curr.weight
                path: list[tuple[int, int]] = []
                while curr != None:
                    path.append((curr.x, curr.y))
                    curr = curr.father
                path.reverse()
                return path, weight
            closed_nodes[key] = curr
            movements: list[tuple[int, int]] = [
                (1, -1),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
            ]
            for idx, move in enumerate(movements):
                m_x = move[0] + curr.x
                m_y = move[1] + curr.y
                next_key = str(m_x) + "," + str(m_y)
                next_weight = curr.weight + int(10 * ((SQRT_2 * ((idx + 1) & 1)) or 1))
                if (
                    next_key in closed_nodes
                    or m_x < 0
                    or m_y < 0
                    or m_y >= len(self.grid)
                    or m_x >= len(self.grid[m_y])
                    or self.grid[m_y][m_x] == 0
                    or (((idx + 1) & 1) and not curr.has_diagonal(*move, self.grid))
                ):
                    continue
                distance = Grid.h_distance((m_x, m_y), destiny)
                if next_key in opened_nodes:
                    if next_weight + distance < opened_nodes[next_key].heuristic:
                        opened_nodes[next_key].heuristic = next_weight + distance
                        opened_nodes[next_key].father = curr
                else:
                    opened_nodes[next_key] = Cell(m_x, m_y, next_weight, curr)

        return [], -1

    def read_file(self, file_name: str) -> None:
        self.grid = [[]]
        with open(file_name, "r") as f:
            f.readline()  # figure type
            h_size = int(f.readline().strip().split(" ")[1])  # height
            f.readline()  # width
            f.readline()  # map
            self.grid = [[]] * h_size
            for index, row in enumerate(f.readlines()):
                self.grid[index] = [1 if x == "." else 0 for x in row.strip()]

    def h_distance(origin: tuple[int, int], destiny: tuple[int, int]) -> int:
        dist_x = abs(origin[0] - destiny[0])
        dist_y = abs(origin[1] - destiny[1])
        return int(10 * (abs(dist_x - dist_y) + min(dist_x, dist_y) * SQRT_2))


class SSG(Grid):
    def __init__(self, grid: list[list] = [[]]) -> None:
        super().__init__(grid)
        self.vertices: dict[str, Vertex] = {}

    def create_graph(self) -> None:
        self.create_verices()
        self.reduce_goals()

        for vertex in self.vertices:
            self.vertices[vertex].create_edges(self.grid)
        for vertex in self.vertices:
            self.vertices[vertex].reduce_edges()

    def create_from_file(self, file_name: str) -> None:
        self.read_file(file_name)
        self.create_graph()

    def create_verices(self) -> None:
        def is_corner(cell: tuple[int, int], dir_x: int, dir_y: int, grid: list[list]):
            x = cell[0]
            y = cell[1]
            pos = (x + dir_x, y + dir_y)
            if (
                x < 0
                or y < 0
                or y >= len(grid)
                or x >= len(grid[y])
                or pos[0] < 0
                or pos[1] < 0
                or pos[1] >= len(grid)
                or pos[0] >= len(grid[pos[1]])
            ):
                return False
            return grid[pos[1]][pos[0]] == 0 and grid[y][pos[0]] == 1 == grid[pos[1]][x]

        self.vertices: dict[str, Vertex] = {}
        for y, row in enumerate(self.grid):
            for x, value in enumerate(row):
                if value == 0:
                    continue
                key = str(x) + "," + str(y)
                cell = (x, y)
                corners: list[tuple[int, int]] = [
                    (-1, -1),  # Superior Esquerda
                    (1, -1),  # Superior Direita
                    (1, 1),  # Inferior Direita
                    (-1, 1),  # Inferior Esquerda
                ]
                for corner in corners:
                    d_x = corner[0]
                    d_y = corner[1]
                    if is_corner(cell, d_x, d_y, self.grid):
                        self.vertices[key] = Vertex(x, y)
                        self.grid[y][x] = self.vertices[key]
                        break
                else:
                    self.grid[y][x] = 1

    def reduce_goals(self) -> None:
        to_delete: list[str] = []
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
        for goal_key in self.vertices:
            if goal_key in to_delete:
                continue
            vertex = self.vertices[goal_key]
            count_neighbors = 0
            for move in movements:
                key = str(move[0] + vertex.x) + "," + str(move[1] + vertex.y)
                if key in self.vertices:
                    count_neighbors += 1
            if count_neighbors > 1:
                self.grid[vertex.y][vertex.x] = 1
                to_delete.append(goal_key)
        self.vertices = dict(
            [(x, self.vertices[x]) for x in self.vertices if x not in to_delete]
        )

    def a_graph_search(
        self, origin: tuple[int, int], destiny: tuple[int, int], skip: list[str] = []
    ) -> tuple[list[Vertex], int]:
        class Cell:
            def __init__(self, vertex: Vertex, weight: int, father: "Cell") -> None:
                self.vertex = vertex
                self.father = father
                self.weight = weight
                self.heuristic = self.weight + Grid.h_distance(
                    (vertex.x, vertex.y), destiny
                )

            def is_destiny(self, target: tuple[int, int]) -> bool:
                return self.vertex.x == target[0] and self.vertex.y == target[1]

            def __lt__(self, other: "Cell") -> bool:
                return self.heuristic < other.heuristic

        key = str(origin[0]) + "," + str(origin[1])
        opened_nodes: dict[str, Cell] = {key: Cell(self.vertices[key], 0, None)}
        closed_nodes: dict[str, Cell] = {}
        while opened_nodes:
            lowest = min(opened_nodes.values())
            key = str(lowest.vertex.x) + "," + str(lowest.vertex.y)
            curr = opened_nodes.pop(key)
            if curr.is_destiny(destiny):
                weight = curr.weight
                path: list[Vertex] = []
                while curr != None:
                    path.append(curr.vertex)
                    curr = curr.father
                path.reverse()
                return path, weight
            closed_nodes[key] = curr
            curr_vertex = curr.vertex
            for edge in curr_vertex.edges:
                curr_edge = curr_vertex.edges[edge]
                next = curr_edge.destiny
                next_key = str(next.x) + "," + str(next.y)
                if next_key in closed_nodes or next_key in skip:
                    continue
                next_weight = curr.weight + curr_edge.weight
                distance = Grid.h_distance((next.x, next.y), destiny)
                if next_key in opened_nodes:
                    if next_weight + distance < opened_nodes[next_key].heuristic:
                        opened_nodes[next_key].heuristic = next_weight + distance
                        opened_nodes[next_key].father = curr
                else:
                    opened_nodes[next_key] = Cell(next, next_weight, curr)

        return [], -1

    def h_reachable(self, origin: "Vertex", destiny: tuple[int, int]) -> bool:
        return (
            Grid.h_distance((origin.x, origin.y), destiny)
            == self.a_graph_search((origin.x, origin.y), destiny)[1]
        )


class TSG(SSG):
    def __init__(self, grid: list[list] = [[]]) -> None:
        super().__init__(grid)
        self.local_goals: dict[str, Vertex] = {}

    def create_from_file(self, file_name: str) -> None:
        self.read_file(file_name)
        self.create_graph()

    def create_graph(self) -> None:
        super().create_graph()
        self.convert_to_tsg()

    def convert_to_tsg(self) -> None:
        global_goals: dict[str, Vertex] = dict(self.vertices)
        # idx = 1
        for s in self.vertices:
            vert = self.vertices[s]
            new_edges: dict[str, tuple[Vertex, Vertex]] = {}
            is_local: bool = True
            visited_edges: set[str] = set()
            # if not idx % int((len(self.vertices) / 5)):
            #     print(f"{idx}/{len(self.vertices)}")
            # idx += 1
            for p in vert.edges:
                for q in vert.edges:
                    if p == q or (q, p) in visited_edges:
                        continue
                    visited_edges.add((p, q))
                    skip = [x for x in self.local_goals if x != p and x != q]
                    skip.append(s)
                    origin = vert.edges[p].destiny
                    destiny = vert.edges[q].destiny
                    _, d = self.a_graph_search(
                        (origin.x, origin.y), (destiny.x, destiny.y), skip
                    )
                    if d == -1 or d > vert.edges[p].weight + vert.edges[q].weight:
                        if self.h_reachable(origin, (destiny.x, destiny.y)):
                            check_key = q + "," + p
                            if check_key not in new_edges:
                                new_edges[p + "," + q] = (origin, destiny)
                        else:
                            is_local = False
                            break
                if not is_local:
                    break
            if is_local:
                for edge in new_edges:
                    edge_tuple = new_edges[edge]
                    edge_tuple[0].add_edge(edge_tuple[1])
                self.local_goals[s] = vert
        self.vertices = dict(
            [(x, global_goals[x]) for x in global_goals if x not in self.local_goals]
        )
